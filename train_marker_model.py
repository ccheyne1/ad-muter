"""Fit glyph prototypes from reviewed development annotations, never held-out images."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from marker_model import proposals,glyph,descriptor,complete_word


def augmented_descriptors(ink,translate=True):
    """Small residual alignment variation after card rectification; no new scenes."""
    for angle in (-3,0,3):
        for dx in (-2,0,2) if translate else (0,):
            matrix=cv2.getRotationMatrix2D((48,32),angle,1)
            matrix[0,2]+=dx
            moved=cv2.warpAffine(ink,matrix,(96,64),flags=cv2.INTER_LINEAR,borderValue=0)
            yield descriptor(moved)


def synthetic_negatives():
    vectors=[]
    for font in (cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_PLAIN):
        for word in ('Add','bad','Ad123','Advertisement','Ac','AC','AA','Au','Ai','CD','00','1:05','0:30','a','d','A'):
            scale=1.5
            size=cv2.getTextSize(word,font,scale,2)[0]
            scale*=min(1,165/max(1,size[0]));size=cv2.getTextSize(word,font,scale,2)[0]
            patch=np.full((80,200,3),205,np.uint8)
            cv2.putText(patch,word,((200-size[0])//2,(80+size[1])//2),font,scale,(15,15,15),2,cv2.LINE_AA)
            for sigma in (0,.8,1.5):
                blurred=patch if not sigma else cv2.GaussianBlur(patch,(5,5),sigma)
                ink=glyph(blurred)
                if ink is not None:vectors.extend(augmented_descriptors(ink,translate=False))
    return vectors


def fit(dataset,index,annotations,out,threshold=.67,margin=.08):
    rows={row['file']:row for row in index['rows']}
    dataset=dataset.resolve()
    def training_frame(row):
        path=(dataset/row['file']).resolve()
        if not path.is_relative_to(dataset):raise ValueError('Training path leaves dataset')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Training source changed')
        frame=cv2.imread(str(path))
        if frame is None or (frame.shape[1],frame.shape[0])!=(row['width'],row['height']):raise ValueError('Invalid training image')
        return frame
    positives=[];positive_batches=[];negatives=[];positive_examples=0;training_files=set()
    target=np.float32([[0,0],[199,0],[199,79],[0,79]])
    for annotation in annotations:
        row=rows[annotation['file']]
        if row['partition']!='development' or row['label']!='ad':raise ValueError('Training annotation leaks outside positive development set')
        frame=training_frame(row);quad=np.float32(annotation['quad'])
        patch=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(quad,target),(200,80))
        ink=glyph(patch)
        if ink is None or not complete_word(ink):continue
        descriptors=list(augmented_descriptors(ink));positive_examples+=1
        training_files.add(row['file'])
        positives.extend(descriptors);positive_batches.extend([row['batch']]*len(descriptors))
    for row in rows.values():
        if row['partition']!='development' or row['label']!='no_ad':continue
        for candidate in proposals(training_frame(row)):
            ink=glyph(candidate.patch)
            if ink is not None:negatives.append(descriptor(ink))
    real_negatives=len(negatives);negatives.extend(synthetic_negatives())
    p=np.array(positives,np.float32);n=np.array(negatives,np.float32)
    if len(p)<5 or len(n)<10:raise ValueError('Insufficient complete glyph examples')
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,templates=p,negatives=n,threshold=np.float32(threshold),margin=np.float32(margin),feature_version=np.int32(3))
    # Report batch-level leave-out similarity, not resubstitution as accuracy.
    leave_out=[]
    for i,vector in enumerate(p):
        other=np.array([batch!=positive_batches[i] for batch in positive_batches])
        if other.any():leave_out.append(float((p[other]@vector).max()))
    provenance={'feature_version':3,'training_partition':'development','positive_examples':positive_examples,
                'positive_templates':len(p),'positive_batches':len(set(positive_batches)),
                'real_negative_glyphs':real_negatives,'synthetic_negative_glyphs':len(n)-real_negatives,
                'threshold':threshold,'margin':margin,'min_leave_batch_out_similarity':min(leave_out),
                'training_files':sorted(training_files),
                'index_sha256':hashlib.sha256(json.dumps(index,sort_keys=True).encode()).hexdigest(),
                'scope':'Hulu badge typography only; no background pixels or camera coordinates are model inputs.'}
    out.with_suffix('.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('dataset'))
    parser.add_argument('--index',type=Path,default=Path('diagnostics/labeled/index.json'))
    parser.add_argument('--annotations',type=Path,default=Path('diagnostics/labeled/marker-annotations.json'))
    parser.add_argument('--out',type=Path,default=Path('models/hulu-marker-v1.npz'))
    args=parser.parse_args()
    print(json.dumps(fit(args.dataset,json.loads(args.index.read_text()),json.loads(args.annotations.read_text()),args.out)))


if __name__=='__main__':main()
