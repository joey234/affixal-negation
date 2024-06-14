import tiktoken
import pandas as pd
from transformers import AutoTokenizer


# Load dictionary
df = pd.read_csv('annotations/all_annotations.txt', sep = '\t')


byte_level_bpe = ["gpt2", "p50k_base", "cl100k_base", 'roberta-base']

sentencepiece = ['xlnet-base-cased', 't5-base', 'meta-llama/Llama-2-7b-hf', 'albert-base-v2']

wordpiece = ['bert-base-uncased', 'google/electra-base-discriminator']


gpt_tokenizers = ["gpt2", "p50k_base", "cl100k_base"]
hf_tokenizers = ['roberta-base'] + sentencepiece+wordpiece


df_affixal = df.loc[ (df['affixal/non-affixal_ann1']=='affixal') | (df['affixal/non-affixal_ann2']=='affixal') ]


negation_words = []
negative_affixes = []


for i, row in df_affixal.iterrows():
    negation_words.append(row['neg_element'])
    if row['affixal/non-affixal_ann1']== 'affixal':
        negative_affixes.append(row['negative affix_ann1'].strip('-'))
    else:
        negative_affixes.append(row['negative affix_ann2'].strip('-'))


def correct_tokenization(token_bytes, negative_affix, negation_word):
    for token in token_bytes:
        # token = str(token, 'UTF-8')
        # print(token)
        if negative_affix == token or token == negation_word:
            return True
    return False


def compare_encodings(example_string, tokenized_dict, negative_affix):
    """Prints a comparison of three string encodings."""
    # print the example string
    print(f'\nExample string: "{example_string}"')
    # for each encoding, print the # of tokens, the token integers, and the token bytes
    # gpt-base
    tokenized_dict[example_string]['negative_affix'] = negative_affix

    for encoding_name in gpt_tokenizers:
        tokenized_dict[example_string][encoding_name] = {'bytes': [], 'correct': False}
        encoding = tiktoken.get_encoding(encoding_name)
        token_integers = encoding.encode(example_string)
        num_tokens = len(token_integers)
        token_bytes = [str(encoding.decode_single_token_bytes(token), 'UTF-8') for token in token_integers]
        # token_bytes = [encoding.decode(token) for token in token_integers]
        tokenized_dict[word][encoding_name]['bytes'] = token_bytes
        tokenized_dict[word][encoding_name]['correct'] = correct_tokenization(token_bytes, negative_affix, example_string)
        # print(correct_tokenization(token_bytes, negative_affix))
        # print()
        # print(f"{encoding_name}: {num_tokens} tokens")
        # print(f"token integers: {token_integers}")
        # print(f"token bytes: {token_bytes}")
    # hf-base
    for encoding_name in hf_tokenizers:
        tokenized_dict[example_string][encoding_name] = {'bytes': [], 'correct':[]}

        tokenizer = AutoTokenizer.from_pretrained(encoding_name)
        token_integers = tokenizer(example_string).input_ids
        num_tokens = len(token_integers)
        # token_bytes= tokenizer.decode(token_integers, skip_special_tokens = True)
        # token_bytes = [tokenizer.token_to_word(token) for token in token_integers]
        token_bytes = [tokenizer.decode(token, skip_special_tokens = True, clean_up_tokenization_spaces = True).strip('#') for token in token_integers]
        token_bytes = list(filter(None, token_bytes))
        tokenized_dict[word][encoding_name]['bytes'] = token_bytes
        tokenized_dict[word][encoding_name]['correct'] = correct_tokenization(token_bytes, negative_affix, example_string)
        # print(correct_tokenization(token_bytes, negative_affix))
        # print()
        # print(f"{encoding_name}: {num_tokens} tokens")
        # print(f"token integers: {token_integers}")
        # print(f"token bytes: {token_bytes}")


tokenized_dict = {}
for i,word in enumerate(negation_words):
    # print(word, negative_affixes[i])
    tokenized_dict[word] = {}
    compare_encodings(word, tokenized_dict, negative_affixes[i])


items = []
for key, value in tokenized_dict.items():
    for tokenizer in byte_level_bpe:
        item = {}
        item['word'] = key
        item['negative_affix'] = value['negative_affix']
        item['tokenizer'] = tokenizer
        item['bytes'] = value[tokenizer]['bytes']
        item['correct'] = value[tokenizer]['correct']
        item['tokenizer_type'] = 'byte_level_bpe'
        items.append(item)
    for tokenizer in sentencepiece:
        item = {}
        item['word'] = key
        item['negative_affix'] = value['negative_affix']
        item['tokenizer'] = tokenizer
        item['bytes'] = value[tokenizer]['bytes']
        item['correct'] = value[tokenizer]['correct']
        item['tokenizer_type'] = 'sentencepiece'
        items.append(item)
    for tokenizer in wordpiece:
        item = {}
        item['word'] = key
        item['negative_affix'] = value['negative_affix']
        item['tokenizer'] = tokenizer
        item['bytes'] = value[tokenizer]['bytes']
        item['correct'] = value[tokenizer]['correct']
        item['tokenizer_type'] = 'wordpiece'
        items.append(item)
    # item = {}
    # item['word'] = key
    # item['negative_affix'] = value['negative_affix']
    # items.append(item)
    # for tokenizer in hf_tokenizers:
    #     item = {}
    #     item['word'] = key
    #     item['negative_affix'] = value['negative_affix']
    #     item['tokenizer'] = tokenizer
    #     item['bytes'] = value[tokenizer]['bytes']
    #     item['correct'] = value[tokenizer]['correct']
    #     items.append(item)

df_tokenized = pd.DataFrame(data = items)
df_tokenized.to_csv('tokenized_data.csv')


