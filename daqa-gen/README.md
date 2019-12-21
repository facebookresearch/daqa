# Diagnostic Audio Question Answering (DAQA)

Code to generate the DAQA dataset as described in [Temporal Reasoning via Audio Question Answering](https://arxiv.org/abs/1911.09655).

<p align="center"><img width="100%" src="../assets/daqa.png" /></p>

The generation process comprises two steps: (1) generate audio clips and descriptions; then (2) generate questions and answers.

# Setup

**Requirements.** Make sure the requirements listed in the parent directory are installed.
[See this tutorial for detailed instructions](https://docs.python.org/3/tutorial/venv.html). You can set up a virtual environment as follows.

```bash
virtualenv -p python3 .env # Create virtual environment
source .env/bin/activate # Activate virtual environment
pip install -r requirements.txt # Install dependencies
# Get things done.
deactivate # Exit virtual environment
```

**Audio Files.** There are two types of audio clips: (1) recorded audio; and (2) audio from [AudioSet](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45857.pdf).

Download the recoded audio.

```bash
wget https://dl.fbaipublicfiles.com/daqa/daqa-audio.tar.gz
```

The youtube IDs and metadata as described in [AudioSet](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45857.pdf) are listed in `daqa_sources.json`.
Note that `daqa_sources.json` contains the youtube IDs as well as the list of the recorded audio.
These are differentiated by using the key `url` for youtube IDs and `dir` for recorded audio.

**Symlinks.** Set some symlinks.

```bash
daqa_gen=../daqa-gen
daqa_gen_local=../daqa-gen-local
daqa_dir=../daqa-dataset
```

# Generate Audio

The following three commands should be used to generate the audio clips for the training, validation, and test sets respectively.

```bash
python3 $daqa_gen/generate_audio.py \
  --seed 0 \
  --num_audio 80000 \
  --set train \
  --dataset $daqa_gen/daqa.json \
  --events $daqa_gen_local/events \
  --backgrounds $daqa_gen_local/backgrounds \
  --output_audio_dir $daqa_dir/audio/train/ \
  --output_narrative_dir $daqa_dir/narratives/train/ \
  --output_narrative_file $daqa_dir/daqa_train_narratives.json
```

```bash
python3 $daqa_gen/generate_audio.py \
  --seed 1 \
  --num_audio 10000 \
  --set val \
  --dataset $daqa_gen/daqa.json \
  --events $daqa_gen_local/events \
  --backgrounds $daqa_gen_local/backgrounds \
  --output_audio_dir $daqa_dir/audio/val/ \
  --output_narrative_dir $daqa_dir/narratives/val/ \
  --output_narrative_file $daqa_dir/daqa_val_narratives.json
```

```bash
python3 $daqa_gen/generate_audio.py \
  --seed 2 \
  --num_audio 10000 \
  --set test \
  --dataset $daqa_gen/daqa.json \
  --events $daqa_gen_local/events \
  --backgrounds $daqa_gen_local/backgrounds \
  --output_audio_dir $daqa_dir/audio/test/ \
  --output_narrative_dir $daqa_dir/narratives/test/ \
  --output_narrative_file $daqa_dir/daqa_test_narratives.json
```

# Generate Questions and Answers

The following three commands should be used to generate the questions and answers for the training, validation, and test sets respectively.

```bash
python3 $daqa_gen/generate_questions_answers.py \
  --seed 0 \
  --dataset $daqa_gen/daqa.json \
  --input_narrative_file $daqa_dir/daqa_train_narratives.json \
  --set train \
  --num_questions_per_narrative 5 \
  --output_qa_file $daqa_dir/daqa_train_questions_answers_5.json
```

```bash
python3 $daqa_gen/generate_questions_answers.py \
  --seed 1 \
  --dataset $daqa_gen/daqa.json \
  --input_narrative_file $daqa_dir/daqa_val_narratives.json \
  --set val \
  --num_questions_per_narrative 10 \
  --output_qa_file $daqa_dir/daqa_val_questions_answers.json
```

```bash
python3 $daqa_gen/generate_questions_answers.py \
  --seed 2 \
  --dataset $daqa_gen/daqa.json \
  --input_narrative_file $daqa_dir/daqa_test_narratives.json \
  --set test \
  --num_questions_per_narrative 10 \
  --output_qa_file $daqa_dir/daqa_test_questions_answers.json
```

# License

Code is released under the CC-BY 4.0 license. See [LICENSE](LICENSE) for additional details.

# Citation

If you find this code useful in your research, please cite:

```bibtex
@inproceedings{fayek2019temporal,
  title = {Temporal Reasoning via Audio Question Answering},
  author = {Haytham M. Fayek and Justin Johnson},
  year = {2019},
}
```
