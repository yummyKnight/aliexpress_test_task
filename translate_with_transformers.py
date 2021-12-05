import torch
from sklearn import datasets
import re
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

AMOUNT_OF_STR = 100


def prepare_ds() -> list:
    newsgroups_train = datasets.fetch_20newsgroups(data_home='data', subset='train', shuffle=True,
                                                   remove=('headers', 'footers', 'quotes'), random_state=24)
    i = 0
    processed_list = []
    for sentence in newsgroups_train.data:
        if len(sentence) > 1024:  # limitations of model
            continue
        processed_list.append(re.sub('\s+', ' ', sentence))
        i += 1
        if i >= AMOUNT_OF_STR:
            break
    return processed_list


def translate(processed_list: list, step: int = 10) -> list:
    mname = "facebook/wmt19-en-ru"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname).cuda()
    last_i = 0
    decoded = []
    for i in range(step, len(processed_list) + step, step):
        out_tensor = tokenizer(processed_list[last_i:i], padding=True, return_tensors="pt")['input_ids'].cuda()
        outputs = model.generate(out_tensor)
        decoded.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        del out_tensor
        torch.cuda.empty_cache()
        last_i = i
    return decoded


def write_to_file(orig: list, translated: list):
    with open("output.txt", 'w', encoding='utf-8') as f:
        for orig_phrase, translated_phrase in zip(orig, translated):
            f.write('Original sentence:\n')
            f.write(orig_phrase)
            f.write('\n')
            f.write('Translated sentence:\n')
            f.write(translated_phrase)
            f.write('\n')


def main():
    processed_list = prepare_ds()
    translated = translate(processed_list)
    write_to_file(processed_list, translated)


if __name__ == '__main__':
    main()
