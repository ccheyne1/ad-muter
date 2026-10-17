import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from collect_frames import save_frame


class DatasetTests(unittest.TestCase):
    def test_manual_labels_append_with_original_dimensions_and_distinct_files(self):
        root=Path(__file__).parent/'diagnostics'/'test-runs';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            out=Path(directory)
            frame=np.full((100,200,3),120,np.uint8)
            first=save_frame(frame,out,'ad',batch='one')
            second=save_frame(frame,out,'no_ad',batch='two')
            rows=[json.loads(line) for line in (out/'manifest.jsonl').read_text().splitlines()]
            self.assertEqual([row['label'] for row in rows],['ad','no_ad'])
            self.assertEqual([row['label_source'] for row in rows],['human','human'])
            self.assertNotEqual(first['file'],second['file'])
            for row in rows:
                self.assertEqual(cv2.imread(str(out/row['file'])).shape,(100,200,3))
                self.assertEqual((row['width'],row['height']),(200,100))
            with self.assertRaises(ValueError):save_frame(frame,out,'guess')
            self.assertEqual(len((out/'manifest.jsonl').read_text().splitlines()),2)


if __name__=='__main__':unittest.main()
