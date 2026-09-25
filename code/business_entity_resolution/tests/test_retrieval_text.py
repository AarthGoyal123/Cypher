import pytest
import pandas as pd
from blocking.retrieval_text import build_retrieval_text, char_ngrams

def test_build_retrieval_text_standard():
    row = pd.Series({
        'business_name_clean': 'apple inc', 
        'business_address_clean': '1 infinite loop'
    })
    expected = "business name: apple inc address: 1 infinite loop"
    assert build_retrieval_text(row) == expected

def test_build_retrieval_text_missing_address():
    row = pd.Series({
        'business_name_clean': 'apple inc', 
        'business_address_clean': float('nan')
    })
    expected = "business name: apple inc address: "
    assert build_retrieval_text(row) == expected

def test_build_retrieval_text_deterministic():
    row1 = pd.Series({'business_name_clean': 'a', 'business_address_clean': 'b'})
    row2 = pd.Series({'business_name_clean': 'a', 'business_address_clean': 'b'})
    assert build_retrieval_text(row1) == build_retrieval_text(row2)

def test_char_ngrams_standard():
    text = "apple"
    expected = {'app', 'ppl', 'ple'}
    assert char_ngrams(text, 3) == expected

def test_char_ngrams_short_string():
    text = "hi"
    expected = {'hi'}
    assert char_ngrams(text, 3) == expected
