"""Evaluation protocol tests: invalid answers, causal scoring, and loss masking."""
from types import SimpleNamespace
import unittest

import torch
from sklearn.metrics import f1_score

from smarthome_efficiency.evaluation import (
    classification_metrics, normalize_modes, parse_generated_label, prediction_agreement,
)
from smarthome_efficiency.model import PilotModel


class EvaluationProtocolTests(unittest.TestCase):
    labels = ['neutral', 'angry', 'happy']

    def test_parser_accepts_only_documented_single_option_grammar(self):
        for text in ('B', ' b ', 'B.', '(b)', 'Answer: B', 'Option (B)!', '\nAnswer:\nB.\n'):
            with self.subTest(text=text):
                self.assertEqual(parse_generated_label(text, self.labels),
                                 {'predicted_label': 'angry', 'invalid_reason': None})
        for text, reason in [('', 'empty_output'), ('Z', 'option_out_of_range'),
                             ('A or B', 'ambiguous_options'), ('B because angry', 'unsupported_format'),
                             ('angry', 'unsupported_format'), ('probably B', 'unsupported_format'),
                             ('B. A.', 'ambiguous_options'), ('banana', 'unsupported_format'), ('AnswerB', 'unsupported_format')]:
            with self.subTest(text=text):
                self.assertEqual(parse_generated_label(text, self.labels),
                                 {'predicted_label': None, 'invalid_reason': reason})

    def test_invalid_outputs_remain_errors_and_false_negatives(self):
        labels = ['a', 'b']
        truth = ['a', 'a', 'b', 'b']
        predictions = ['a', None, 'b', 'unrecognized']
        result = classification_metrics(truth, predictions, labels)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['invalid_count'], 2)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 1], [0, 1, 1]])
        self.assertEqual(result['confusion_columns'], ['a', 'b', '<invalid>'])
        self.assertAlmostEqual(result['weighted_f1'], 2 / 3)
        self.assertAlmostEqual(result['macro_f1'], 2 / 3)
        self.assertEqual(result['per_class']['a']['recall'], 0.5)
        self.assertEqual(result['per_class']['a']['precision'], 1.0)
        normalized = [label if label in labels else '<invalid>' for label in predictions]
        for average in ('weighted', 'macro'):
            self.assertAlmostEqual(result[f'{average}_f1'], f1_score(
                truth, normalized, labels=labels, average=average, zero_division=0))

    def test_metrics_include_missing_classes_and_reject_unknown_ground_truth(self):
        result = classification_metrics(['a'], [None], ['a', 'b'])
        self.assertEqual(result['accuracy'], 0.0)
        self.assertEqual(result['weighted_f1'], 0.0)
        self.assertEqual(result['per_class']['b']['support'], 0)
        with self.assertRaises(ValueError):
            classification_metrics(['unknown'], ['a'], ['a', 'b'])
        with self.assertRaises(ValueError):
            classification_metrics(['a'], [], ['a', 'b'])

    def test_agreement_does_not_discard_invalid_greedy_predictions(self):
        result = prediction_agreement(['a', 'b', 'a'], ['a', None, 'b'])
        self.assertEqual(result['agreement_count'], 1)
        self.assertAlmostEqual(result['agreement'], 1 / 3)
        self.assertEqual(result['greedy_invalid_count'], 1)
        self.assertEqual(result['agreement_on_valid'], 0.5)

    def test_modes_are_explicit_and_stably_ordered(self):
        self.assertEqual(normalize_modes('both'), ('logit', 'greedy'))
        self.assertEqual(normalize_modes(['greedy', 'logit', 'greedy']), ('logit', 'greedy'))
        self.assertEqual(normalize_modes('greedy'), ('greedy',))
        for value in ([], 'generation', ['logit', 'unknown']):
            with self.assertRaises(ValueError):
                normalize_modes(value)


class CausalToyModel(torch.nn.Module):
    """Small genuinely causal forward, with captured inputs and deterministic generation."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.rope_deltas = None
        self.last_labels = None
        self.last_input_ids = None
        self.generation_kwargs = None
        self.generate_ids = [7, 9]

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

    def get_base_model(self):
        return self

    def forward(self, input_ids, labels=None, **kwargs):
        self.last_input_ids = input_ids.detach().clone()
        self.last_labels = labels.detach().clone() if labels is not None else None
        prefix_sum = input_ids.float().cumsum(dim=1)
        vocabulary = torch.linspace(-0.5, 0.5, 16)
        logits = prefix_sum.unsqueeze(-1) * vocabulary * self.weight
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1), ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, input_ids, **kwargs):
        self.generation_kwargs = kwargs
        self.generation_input = input_ids.detach().clone()
        return torch.cat((input_ids, torch.tensor([self.generate_ids])), dim=1)


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.runner = PilotModel.__new__(PilotModel)
        self.runner.model = CausalToyModel()
        self.runner.labels = ['negative', 'positive']
        self.runner.label_token_ids = [5, 7]
        self.runner.eos_token_id = 9
        self.runner.lora_enabled = True
        self.runner.processor = SimpleNamespace(tokenizer=SimpleNamespace(
            pad_token_id=0,
            decode=lambda ids, skip_special_tokens: ''.join({5: 'A', 7: 'B', 9: '', 11: '?'}[i] for i in ids),
        ))
        self.batch = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)}

    def test_mask_and_shift_supervise_exactly_class_and_eos(self):
        result = self.runner.loss_batch(self.batch, 1)
        self.assertEqual(self.runner.model.last_input_ids.tolist(), [[1, 2, 3, 7, 9]])
        self.assertEqual(self.runner.model.last_labels.tolist(), [[-100, -100, -100, 7, 9]])
        self.assertAlmostEqual(result['loss'].item(), ((result['class_loss'] + result['eos_loss']) / 2).item(), places=6)
        self.assertTrue(result['loss'].requires_grad)
        self.assertFalse(result['class_loss'].requires_grad)
        self.assertFalse(result['eos_loss'].requires_grad)
        result['loss'].backward()
        self.assertNotEqual(self.runner.model.weight.grad.item(), 0)
        self.assertEqual(self.batch['input_ids'].tolist(), [[1, 2, 3]])

    def test_teacher_forced_choice_logits_equal_prompt_only_and_ignore_gold_choice(self):
        prompt_scores = self.runner.score(self.batch)
        first = self.runner.evaluate_batch(self.batch, 0, 'logit')
        second = self.runner.evaluate_batch(self.batch, 1, ['logit'])
        self.assertEqual(first['logit_scores'], prompt_scores)
        self.assertEqual(second['logit_scores'], prompt_scores)
        self.assertIsInstance(first['sft_loss'], float)
        self.assertNotIn('generated_text', first)

    def test_greedy_receives_no_gold_answer_and_has_no_choice_restriction(self):
        result = self.runner.evaluate_batch(self.batch, 0, 'both')
        self.assertEqual(self.runner.model.generation_input.tolist(), [[1, 2, 3]])
        self.assertEqual(result['generated_text'], 'B')
        self.assertEqual(result['greedy_predicted_label'], 'positive')
        self.assertEqual(result['generated_token_ids'], [7, 9])
        kwargs = self.runner.model.generation_kwargs
        self.assertFalse(kwargs['do_sample'])
        self.assertEqual(kwargs['num_beams'], 1)
        self.assertTrue(kwargs['use_cache'])
        self.assertEqual(kwargs['eos_token_id'], 9)
        self.assertNotIn('prefix_allowed_tokens_fn', kwargs)
        self.assertNotIn('logits_processor', kwargs)
        self.runner.model.generate_ids = [11, 9]
        invalid = self.runner.evaluate_batch(self.batch, 0, 'both')
        self.assertIsNone(invalid['greedy_predicted_label'])
        self.assertIsNotNone(invalid['invalid_reason'])
        self.assertIsNotNone(invalid['logit_predicted_label'])

    def test_training_can_forward_through_wrapper_without_replacing_saved_model(self):
        underlying = self.runner.model
        calls = []
        class ForwardWrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, **kwargs):
                calls.append(True)
                return self.module(**kwargs)
        wrapper = ForwardWrapper(underlying)
        loss = self.runner.train_batch(self.batch, 1, forward_model=wrapper)
        loss.backward()
        self.assertEqual(calls, [True])
        self.assertIs(self.runner.model, underlying)
        self.assertIsNotNone(underlying.weight.grad)


if __name__ == '__main__':
    unittest.main()
