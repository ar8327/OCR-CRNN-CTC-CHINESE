# OCR-CRNN-CTC

### This repo contains codes that is refactored & improved version of https://github.com/Li-Ming-Fan/OCR-CRNN-CTC

### Improvements : 
+ Fixed some bug in data loading
+ Added support for Chinese
+ Added tools to generate random number dictionaries
+ Added noises(lines,etc) to sample generator
+ Added random color selection to sample generator
+ Reduced checkpoint size by only saving trainable variables

### Fast example
+ Get some fonts and change `list_fonts` in 'tools/data_generator.py'
+ Generate dictionary by `running tools/gen_numbers.py`
+ Generate samples by running `tools/data_generator.py` and cut down text samples by running `tools/data_rects_extractor.py`
+ Change path of 'data_rects_train' and 'data_rects_valid' in train.py
+ Run train.py and you're ready to go


### Train a Chinese-enabled model
+ Change `NUMBER_ONLY` in `core/meta.py` to `False`
+ Get some Chinese fonts (like Microsoft YaHei) and change `list_fonts` in 'tools/data_generators.py'
+ Change `words_file` in 'tools/data_generator.py' to your dictionary file
+ Follow steps in Fast example to generate some samples
+ You may consider change `beam_width` to 20 in order to get a better overall prediction accuracy.

### About other files in tools/
+ 'tools/dictionary_cut.py' is for enhancing samples in dictionaries. It will cut a single word many times in order to increase number of words in the dictionary.
+ 'tools/merge_dictionaries.py' is for merging dictionaries into one.

### Prediction
+ Please see `test.py` for prediction example