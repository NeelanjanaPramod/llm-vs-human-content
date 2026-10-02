# LLM vs Human Text Detection (HC3)

Binary text classifier: was an answer written by a human or by ChatGPT? Classical TF-IDF models plus a stronger gradient-boosting stack, evaluated on a held-out test set split by question.

## Dataset
- HC3 (Hello-SimpleAI/HC3, public on Hugging Face). The notebook downloads it automatically: https://huggingface.co/datasets/Hello-SimpleAI/HC3
- 24,322 questions from 5 sources: reddit_eli5, finance, medicine, open_qa, wiki_csai.
- Raw answers: 58,546 human and 26,903 ChatGPT.
- Subsample used: 9,000 random questions (to keep Colab runtime short). After cleaning: 30,483 answers, 20,445 human and 10,038 ChatGPT (33% ChatGPT).

## Pipeline
1. Download HC3 and flatten to one row per answer with a human/ChatGPT label.
2. Clean: collapse whitespace, truncate to 1,500 characters, drop texts under 20 characters, drop duplicate texts.
3. Subsample 9,000 questions.
4. Split 80/20 grouped by question, so no question appears in both train and test (asserted in code). Train 24,352, test 6,131.
5. Features: word TF-IDF (1-2 grams) and character TF-IDF (3-4 grams), 139k features in total, plus 16 style features (length, punctuation rates, and similar).
6. Models: Logistic Regression, Linear SVM, Naive Bayes, Random Forest (style features only), a LightGBM stack (style features + SVD + out-of-fold Logistic Regression score from grouped CV), and an average of LR + SVM + LightGBM.

## Results (held-out test)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.9931 | 0.9896 | 0.9997 |
| Linear SVM | 0.9938 | 0.9906 | 0.9998 |
| Naive Bayes | 0.9392 | 0.9045 | 0.9827 |
| Random Forest (style only) | 0.9705 | 0.9554 | 0.9954 |
| LightGBM stack | 0.9972 | 0.9958 | 0.9999 |
| Average (LR + SVM + LGBM) | 0.9974 | 0.9960 | 0.9998 |

LightGBM accuracy by source: reddit_eli5 0.999, finance 0.992, open_qa 0.987, medicine 1.0, wiki_csai 0.974.

Cross-domain check: train on eli5 + finance + open_qa, test on unseen medicine + wiki_csai (1,638 answers), Logistic Regression: accuracy 0.917, F1 0.924, ROC-AUC 0.991.

## Caveats
- HC3 has a single generator (ChatGPT, late 2022). It is a known easy dataset, so ~99% in-distribution is expected.
- Accuracy drops to about 92% on unseen domains.
- Not tested on newer LLMs, other generators, paraphrased or human-edited text, or adversarial text. This is not evidence that the model detects LLM text in general.
- Trained on a 9,000-question subsample, not the full dataset.

## Run
Open `LLM_vs_Human_HC3.ipynb` in Google Colab and use Runtime > Run all. No Drive mount and no API keys are needed.
