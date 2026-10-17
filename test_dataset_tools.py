import unittest
from dataset_tools import freeze_split,metrics


class EvaluationTests(unittest.TestCase):
    def test_unknowns_are_not_removed_from_recall(self):
        rows=[{'label':label,'ad':value,'processing_seconds':.5} for label,value in
              [('ad',True),('ad',None),('no_ad',False),('no_ad',None)]]
        report=metrics(rows)
        self.assertEqual(report['ad_recall_including_unknowns'],.5)
        self.assertEqual(report['no_ad_recall_including_unknowns'],.5)
        self.assertEqual(report['coverage'],.5)

    def test_split_keeps_batches_together_and_requires_both_labels(self):
        from datetime import datetime,timedelta,timezone
        origin=datetime(2020,1,1,tzinfo=timezone.utc)
        groups=[('no_ad',0),('no_ad',1),('no_ad',2),('ad',3),
                ('no_ad',5),('no_ad',6),('no_ad',7),('ad',20),('no_ad',21)]
        rows=[{'label':label,'batch':f'batch-{i}',
               'captured_at_utc':(origin+timedelta(minutes=minute,seconds=frame)).isoformat()}
              for i,(label,minute) in enumerate(groups) for frame in range(3)]
        split=freeze_split(rows)
        development={r['batch'] for r in split['rows'] if r['partition']=='development'}
        heldout={r['batch'] for r in split['rows'] if r['partition']=='heldout'}
        self.assertFalse(development&heldout)
        for part in ('development','heldout'):
            self.assertEqual({r['label'] for r in split['rows'] if r['partition']==part},{'ad','no_ad'})


if __name__=='__main__':unittest.main()
