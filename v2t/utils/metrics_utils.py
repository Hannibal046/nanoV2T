def eval_v2t(hyps,refs,embedder=None):
    results = {}
    results['bleu'] = round(get_bleu_score(hyps,refs),1)
    results["token_f1"] = round(get_token_f1(hyps,refs)*100,1)
    results['em'] = round(get_exact_match(hyps,refs)*100,1)
    if embedder is not None:
        results['cos_sim'] = round(get_cos_sim(hyps,refs,embedder),2)
    return results

def get_bleu_score(hyps,refs):
    # from: https://github.com/jxmorris12/vec2text/blob/f1369cbfe6bf9e216ac2a3758369c165278c9046/vec2text/trainers/base.py#L335
    import evaluate
    bleu = evaluate.load("sacrebleu")
    bleu_score = [
        bleu.compute(predictions=[p], references=[r])["score"]
        for p, r in zip(hyps, refs)
    ]
    return sum(bleu_score)/len(bleu_score)


def get_token_f1(hyps,refs):
    # from: https://github.com/jxmorris12/vec2text/blob/f1369cbfe6bf9e216ac2a3758369c165278c9046/vec2text/trainers/base.py#L288
    import nltk
    f1s = []
    for hyp,ref in zip(hyps,refs):
        hyp = set(nltk.tokenize.word_tokenize(hyp))
        ref = set(nltk.tokenize.word_tokenize(ref))

        TP = len(ref & hyp)
        FP = len(ref) - len(ref & hyp)
        FN = len(hyp) - len(ref & hyp)

        precision = (TP) / (TP + FP + 1e-20)
        recall = (TP) / (TP + FN + 1e-20)
        try:
            f1 = (2 * precision * recall) / (precision + recall + 1e-20)
        except ZeroDivisionError:
            f1 = 0.0
        f1s.append(f1)
    return sum(f1s)/len(f1s)

def get_exact_match(hyps,refs):
    # from: https://github.com/jxmorris12/vec2text/blob/f1369cbfe6bf9e216ac2a3758369c165278c9046/vec2text/trainers/base.py#L350C9-L350C78
    import numpy as np
    exact_match_score =  (np.array(hyps) == np.array(refs))
    return sum(exact_match_score)/len(exact_match_score)

def get_cos_sim(hyps,refs,embedder):
    # from: https://github.com/jxmorris12/vec2text/blob/f1369cbfe6bf9e216ac2a3758369c165278c9046/vec2text/trainers/base.py#L480
    import torch
    hyp_embeds = embedder.encode(hyps,convert_to_tensor=True,show_progress_bar=False)
    ref_embeds = embedder.encode(refs,convert_to_tensor=True,show_progress_bar=False)
    scores = torch.nn.CosineSimilarity(dim=1)(hyp_embeds, ref_embeds)
    return scores.mean().item()
