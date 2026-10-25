import unittest
from unittest.mock import patch
from urllib.error import URLError
from concrete_audit.evidence_batch import flags, request_metadata


class EvidenceBatchTests(unittest.TestCase):
    def test_zero_cement_needs_other_design_clues_and_remains_pending(self):
        self.assertFalse(flags({'Cem':0,'AlkaliAct':0,'GGBS':550},''))
        result=flags({'Cem':0,'AlkaliAct':189,'GGBS':550},'')
        self.assertEqual(len(result),1);self.assertIn('需PDF确认',result[0])

    def test_title_or_note_alone_does_not_prove_cement_zero(self):
        self.assertFalse(flags({'Cem':None},'alkali-activated underwater concrete'))

    def test_mortar_and_inferred_scope_are_review_hints_not_exclusion(self):
        result=flags({'CastMethod':'Mortar through tube'},'CastingEnv=Underwater属推定')
        self.assertEqual(len(result),2)
        self.assertTrue(all(isinstance(r,str) for r in result))

    def test_proxy_failure_does_not_claim_publisher_status_or_pdf_verification(self):
        with patch('urllib.request.urlopen',side_effect=URLError('Tunnel connection failed: 403 Forbidden')):
            result=request_metadata('https://doi.org/example',1)
        self.assertEqual(result['status'],'request_failed')
        self.assertNotIn('http_status',result);self.assertIn('不能据此推断',result['interpretation'])


if __name__=='__main__':unittest.main()
