""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols, symbols_zh
from config import DataConfig

# Mappings from symbol to numeric ID and vice versa:
if DataConfig.language_model == 'chinese':
  sym = symbols_zh
  cleaner = cleaners.chinese_cleaners1
elif DataConfig.language_model == 'english':
  sym = symbols
  cleaner = cleaners.english_cleaners2
else:
  exit(0)

_symbol_to_id = {s: i for i, s in enumerate(sym)}
_id_to_symbol = {i: s for i, s in enumerate(sym)}


def text_to_sequence(text):
  clean_text = cleaner(text)
  sequence = cleaned_text_to_sequence(clean_text)
  return sequence


def cleaned_text_to_sequence(cleaned_text):
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence


def sequence_to_text(sequence):
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result
