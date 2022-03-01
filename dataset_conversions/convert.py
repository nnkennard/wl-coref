import argparse
import collections
import glob
import json
import stanza
import tqdm

from stanza.utils.conll import CoNLL

STANZA_PIPELINE = stanza.Pipeline('en',
                                  processors='tokenize,lemma,pos,mwt,constituency',
                                  tokenize_pretokenized=True,
                                  tokenize_no_ssplit=True)

parser = argparse.ArgumentParser(
    description=
    'Convert dataset into jsonlines and add syntax info for word level coref.')
parser.add_argument('-i',
                    '--input_format',
                    default="litbank",
                    type=str,
                    help='Input dataset/format')
parser.add_argument('-d', '--data_dir', type=str, help='Data location')

NO_SPEAKER = "no_speaker"


def coref_to_spans(coref_col, offset):
  span_starts = collections.defaultdict(list)
  complete_spans = []
  for i, orig_label in enumerate(coref_col):
    if orig_label == '-':
      continue
    else:
      labels = orig_label.split("|")
      for label in labels:
        if label.startswith("("):
          if label.endswith(")"):
            complete_spans.append((i, i, label[1:-1]))
          else:
            span_starts[label[1:]].append(i)
        elif label.endswith(")"):
          ending_cluster = label[:-1]
          assert len(span_starts[ending_cluster]) in [1, 2]
          maybe_start_idx = span_starts[ending_cluster].pop(-1)
          complete_spans.append((maybe_start_idx, i, ending_cluster))

  span_dict = collections.defaultdict(list)
  for start, end, cluster in complete_spans:
    span_dict[cluster].append((offset + start, offset + end))

  return span_dict


def litbank_extract(filename):
  token_lists = []
  coref_cols = []
  document_id = None
  with open(filename, 'r') as f:
    sentence = []
    for line in f:
      if not line.strip() or line.startswith("#"):
        if sentence:
          tokens, coref_col = zip(*sentence)
          token_lists.append(list(tokens))
          coref_cols.append(coref_col)
          sentence = []
      else:
        fields = line.strip().split()
        maybe_doc_id, sent_idx, tok_idx, token = fields[:4]
        if document_id is None:
          document_id = maybe_doc_id
        else:
          assert maybe_doc_id == document_id
        coref = fields[-1]
        sentence.append((token, coref))

  document_id = "pt/" + document_id
  return document_id, token_lists, coref_cols


RELEVANT_LABELS = "deprel head pos".split()


class WLCorefDocument(object):

  def __init__(self, document_id, tokens, coref_cols=None, clusters=None):
    self.document_id = document_id
    self.tokens = [t for t in tokens if "".join(t).strip()]
    if clusters is not None:
      self.clusters = self._remove_singletons(clusters)
    else:
      assert coref_cols is not None
      self.clusters = self._get_clusters(coref_cols)
    self.syntax = self.get_syntactic_labels()

  def _remove_singletons(self, clusters):
    return [c for c in clusters if len(c) > 1]

  def _get_clusters(self, coref_cols):
    offset = 0
    document_spans = collections.defaultdict(list)
    document_tokens = []
    for coref_col in coref_cols:
      for entity, spans in coref_to_spans(coref_col, offset).items():
        document_spans[entity] += spans
      offset += len(coref_col)
    # Skip singletons
    return self._remove_singletons(document_spans.values())

  def get_syntactic_labels(self):
    offset = 0
    label_collector = {k: [] for k in RELEVANT_LABELS}

    stanza_prepped_doc = "\n\n".join(
    [" ".join(sentence_tokens) for sentence_tokens in
    self.tokens])

    print(stanza_prepped_doc)

    doc = STANZA_PIPELINE(stanza_prepped_doc)

    print(len(doc.sentences))
    print(doc.sentences[0])
    exit()

    for sentence in doc.sentences:
      print(sentence)
      con = sentence.constituency
      for i in str(con).split():
        print(i)
      print("*"*80)
      break
    exit()

    for sentence_tokens in self.tokens:
      if not " ".join(sentence_tokens).strip():
        return None
        assert not labels.sentences[1]
      tokens = labels.sentences[0].tokens
      for token in tokens:
        token_dict, = token.to_dict()
        label_collector['deprel'].append(token_dict['deprel'])
        label_collector['pos'].append(token_dict['xpos'])
        head = token_dict['head']
        if head == 0:
          label_collector['head'].append(None)
        else:
          label_collector['head'].append(head - 1)
      offset += len(sentence_tokens)
    return label_collector

  def produce_jsonl(self):
    if self.syntax is None:
      return None
    sent_id = []
    for i, sent_tokens in enumerate(self.tokens):
      sent_id += [i for _ in sent_tokens]
    flat_tokens = sum(self.tokens, [])
    return json.dumps({
        'document_id': self.document_id,
        'cased_words': flat_tokens,
        'sent_id': sent_id,
        'part_id': 0,
        'speaker': [NO_SPEAKER for _ in flat_tokens],
        'pos': self.syntax['pos'],
        'deprel': self.syntax['deprel'],
        'head': self.syntax['head'],
        'clusters': self.clusters
    })


def process_litbank(data_dir):
  docs = []
  for filename in tqdm.tqdm(glob.glob(data_dir + "/*")):
    document_id, sentences, coref_cols = litbank_extract(filename)
    doc = WLCorefDocument(document_id, sentences, coref_cols=coref_cols)
    docs.append(doc.produce_jsonl())


SUBSETS = "train dev".split()
def process_preco(data_dir):
  docs = {subset:[] for subset in SUBSETS}
  for subset in SUBSETS:
    with open(data_dir + "/" + subset + ".jsonl", 'r') as f:
      for line in tqdm.tqdm(f.readlines()[:500]):
        obj = json.loads(line)
        doc = WLCorefDocument(obj['id'], obj['sentences'], clusters=obj['mention_clusters'])
        docs[subset].append(doc.produce_jsonl())

  for k, v in docs.items():
    print(k, ": ", len([a for a in v if a is None]), "errors")


def main():
  args = parser.parse_args()
  assert args.input_format in ["litbank", "preco"]
  if args.input_format == "litbank":
    process_litbank(args.data_dir)
  elif args.input_format == "preco":
    process_preco(args.data_dir)


if __name__ == "__main__":
  main()

