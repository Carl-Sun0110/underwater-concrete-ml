"""图中明确标注与文本证据分别保留，页码/图像/PDF版本错配必须失败。"""
from pathlib import Path
import tempfile,hashlib
import unittest
from concrete_audit.field_review import validate_evidence

class ImageEvidenceTests(unittest.TestCase):
    def test_image_file_and_pdf_binding_validated(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'figure.png';p.write_bytes(b'fixture image bytes')
            ref={'page':1,'quote':'Fig. 8 Compressive strength','image_path':str(p),
                 'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_pdf_sha256':'pdf-hash'}
            claim={'evidence':[ref]}
            validate_evidence(claim,['Fig. 8 Compressive strength'],'pdf-hash')
            with self.assertRaises(ValueError):validate_evidence(claim,['Fig. 8 Compressive strength'],'other-pdf')
            with self.assertRaises(ValueError):validate_evidence(claim,['Fig. 8 Compressive strength'])

    def test_modified_image_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'figure.png';p.write_bytes(b'original')
            ref={'page':1,'quote':'Fig.6','image_path':str(p),'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_pdf_sha256':'paper'}
            p.write_bytes(b'modified')
            with self.assertRaises(ValueError):validate_evidence({'evidence':[ref]},['Fig.6'],'paper')

    def test_image_does_not_allow_fabricated_text_quote(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'figure.png';p.write_bytes(b'image')
            ref={'page':1,'quote':'134.6 in extracted text','image_path':str(p),
                 'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_pdf_sha256':'paper'}
            with self.assertRaises(ValueError):validate_evidence({'evidence':[ref]},['Fig8 caption only'],'paper')
