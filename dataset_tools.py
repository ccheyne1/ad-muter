"""Audit human labels, freeze a chronological split, and score without dropping abstentions."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

ALIASES={'ad':'ad','ads':'ad','no_ad':'no_ad','no_ads':'no_ad','unknown':'unknown'}


def audit(dataset):
    dataset=Path(dataset).resolve()
    manifest=dataset/'manifest.jsonl'
    source=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows=[];seen={};files=set()
    for row in source:
        label=ALIASES[row['label']]
        path=(dataset/row['file']).resolve()
        if not path.is_relative_to(dataset):raise ValueError('Manifest path leaves dataset')
        if path in files:raise ValueError('Duplicate manifest path: '+row['file'])
        files.add(path)
        checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum!=row['sha256']:raise ValueError('Checksum mismatch: '+row['file'])
        if checksum in seen:raise ValueError('Duplicate image bytes: '+row['file'])
        seen[checksum]=label
        if ALIASES[path.parent.name]!=label:raise ValueError('Folder and label disagree')
        if row.get('label_source')!='human':raise ValueError('Expected human labels')
        if not row.get('batch'):raise ValueError('Missing batch grouping')
        rows.append({**row,'label':label})
    unlisted={p.resolve() for p in dataset.rglob('*.jpg')}-files
    if unlisted:raise ValueError('Unlisted JPEGs in dataset: '+str(len(unlisted)))
    return sorted(rows,key=lambda row:row['captured_at_utc'])


def freeze_split(rows):
    """Last ad break and two preceding negative batches form the untouched split."""
    positives=[r for r in rows if r['label']=='ad']
    breaks=[]
    for row in positives:
        stamp=datetime.fromisoformat(row['captured_at_utc'])
        if not breaks or (stamp-datetime.fromisoformat(breaks[-1][-1]['captured_at_utc'])).total_seconds()>120:
            breaks.append([])
        breaks[-1].append(row)
    if len(breaks)<2:raise ValueError('Need two separate positive breaks for this chronological split')
    last_start=breaks[-1][0]['captured_at_utc']
    negative_batches=list(dict.fromkeys(r['batch'] for r in rows if r['label']=='no_ad' and r['captured_at_utc']<last_start))
    if len(negative_batches)<3:raise ValueError('Need negative batches on both sides of split')
    heldout_negative=set(negative_batches[-2:])
    cutoff=min(r['captured_at_utc'] for r in rows if r['batch'] in heldout_negative)
    indexed=[{**r,'partition':'heldout' if r['captured_at_utc']>=cutoff else 'development'} for r in rows]
    groups=defaultdict(set)
    for row in indexed:groups[row['batch']].add(row['partition'])
    if any(len(partitions)!=1 for partitions in groups.values()):raise ValueError('Batch leakage across split')
    for partition in ('development','heldout'):
        if {r['label'] for r in indexed if r['partition']==partition} != {'ad','no_ad'}:
            raise ValueError('Each partition must contain both labels')
    return {'cutoff_utc':cutoff,'ad_breaks':len(breaks),'rows':indexed,
            'scope':'One camera position/session; last ad break and preceding negative batches held out. Not field accuracy.'}


def metrics(rows):
    tp=sum(r['label']=='ad' and r['ad'] is True for r in rows)
    fp=sum(r['label']=='no_ad' and r['ad'] is True for r in rows)
    tn=sum(r['label']=='no_ad' and r['ad'] is False for r in rows)
    fn=sum(r['label']=='ad' and r['ad'] is False for r in rows)
    ua=sum(r['label']=='ad' and r['ad'] is None for r in rows)
    un=sum(r['label']=='no_ad' and r['ad'] is None for r in rows)
    def divide(a,b):return round(a/b,4) if b else None
    return {'images':len(rows),'true_positive':tp,'false_positive':fp,'true_negative':tn,'false_negative':fn,
            'unknown_ad':ua,'unknown_no_ad':un,'ad_precision':divide(tp,tp+fp),
            'ad_recall_including_unknowns':divide(tp,tp+fn+ua),'no_ad_recall_including_unknowns':divide(tn,tn+fp+un),
            'correct_fraction_including_unknowns':divide(tp+tn,len(rows)),
            'coverage':divide(tp+fp+tn+fn,len(rows)),
            'median_seconds':round(statistics.median(r['processing_seconds'] for r in rows),3) if rows else None,
            'p95_seconds':round(sorted(r['processing_seconds'] for r in rows)[max(0,int(len(rows)*.95)-1)],3) if rows else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('dataset'))
    parser.add_argument('--index',type=Path,default=Path('diagnostics/labeled/index.json'))
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--partition',choices=('development','heldout','all'),default='development')
    parser.add_argument('--out',type=Path,default=Path('diagnostics/labeled/evaluation.json'))
    parser.add_argument('--profile',type=Path,default=Path(__file__).with_name('hulu-profile.json'))
    args=parser.parse_args()
    if args.freeze:
        if args.index.exists():raise ValueError('Frozen split already exists; do not overwrite it after tuning')
        index=freeze_split(audit(args.dataset));args.index.parent.mkdir(parents=True,exist_ok=True)
        args.index.write_text(json.dumps(index,indent=2),encoding='utf-8')
        print(json.dumps({'frozen':str(args.index),'counts':dict(Counter((r['partition']+':'+r['label']) for r in index['rows']))}));return
    import cv2
    from adaptive_vision import Detector
    index=json.loads(args.index.read_text(encoding='utf-8'))
    profile=json.loads(args.profile.read_text(encoding='utf-8'))
    report=[]
    for row in index['rows']:
        if args.partition!='all' and row['partition']!=args.partition:continue
        path=args.dataset/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Dataset changed after split')
        frame=cv2.imread(str(path))
        if frame is None or (frame.shape[1],frame.shape[0])!=(row['width'],row['height']):raise ValueError('Invalid image dimensions')
        # Independent frames avoid carrying geometry or timing state across separate bursts.
        result=Detector(profile).inspect(frame)
        report.append({**row,**result})
        print(json.dumps({'file':row['file'],'label':row['label'],'ad':result['ad'],'reason':result.get('reason'),
                          'processing_seconds':result['processing_seconds']}),flush=True)
    summary={'partition':args.partition,'scope':index['scope'],'metrics':metrics(report),'rows':report}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['metrics']),flush=True)


if __name__=='__main__':main()
