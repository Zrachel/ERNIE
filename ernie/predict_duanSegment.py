# coding=utf-8

import reader.task_reader as task_reader
from model.ernie import ErnieConfig
from finetune.classifier import create_model_predict
from finetune.classifier import evaluate
from optimization import optimization
from utils.args import print_arguments
from utils.init import init_pretraining_params
from utils.init import init_checkpoint
import paddle.fluid as fluid
from finetune_args import parser
import datetime
import csv
import os
import re
import sys
import numpy as np
import pdb

place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
exe = fluid.Executor(place)

startup_prog = fluid.Program()
args = parser.parse_args()
ernie_config = ErnieConfig(args.ernie_config_path)
ernie_config.print_config()
test_prog = fluid.Program()

with fluid.program_guard(test_prog, startup_prog):
    with fluid.unique_name.guard():
        _, graph_vars = create_model_predict(
            args,
            ernie_config=ernie_config,
            is_prediction=True)

LABELMODE = ["0", '1', '2', '3', '4'] # the last class denotes not to duanju
SYMBOLS = [',', '.', '?', '|']

test_prog = test_prog.clone(for_test=True)
exe.run(startup_prog)

init_checkpoint(
    exe,
    args.init_checkpoint,
    main_program=startup_prog,
    use_fp16=args.use_fp16)

reader = task_reader.ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed)

print(datetime.datetime.now())


class Example(object):
    """THIS IS A DOCSTRING"""
    def __init__(self, texta, textb=None):
        """THIS IS A DOCSTRING"""
        if textb is None:
            self._fields = ["text_a", "label"]
        else:
            self._fields = ['text_a', 'text_b', 'label']
            self.text_b = textb
        self.text_a = texta
        self.label = 1


class Sample(object):
    """THIS IS A DOCSTRING"""
    def __init__(self, values):
        """THIS IS A DOCSTRING"""
        #self.names = ['read_file_0.tmp_0', 'read_file_0.tmp_1', 'read_file_0.tmp_2', 'read_file_0.tmp_3']
        self.names = ['eval_placeholder_0', 'eval_placeholder_1', 'eval_placeholder_2', 'eval_placeholder_4', 'eval_placeholder_3']
        self.values = values
        #self.values = [[val] for val in values]

    def gen(self):
        """THIS IS A DOCSTRING"""
        sample = dict(zip(self.names, self.values[0:5]))
        return sample


def predict_oneline(line):
    """THIS IS A DOCSTRING"""
    texta, textb = line
    if textb is None:
        example = Example(texta)
    else:
        example = Example(texta, textb)
    data_gen = reader._prepare_one_sample(example, 1)
    sample = Sample(data_gen)
    sample = sample.gen()

    #print("Final test result:")
    fetch_list = [graph_vars["probs"].name]

    np_probs = exe.run(program=test_prog, feed=sample, fetch_list=fetch_list, use_program_cache=True)
    np_probs = np_probs[0][0]
    #print(datetime.datetime.now())
    #print(np_probs)
    return np_probs


class FLAGS(object):
    """THIS IS A DOCSTRING"""
    max_wait_words = 20
    #waitn = 3
    #appendn = 0
    #textb_none = True
    waitn = 0
    appendn = 3
    textb_none = False


def get_periods_position(sent):
    """THIS IS A DOCSTRING"""
    sent = sent.strip().lower()
    sent = re.sub(r"\"|- |:", "", sent)
    sent = re.sub("ph.d.", "phd", sent)
    #tknzr = TweetTokenizer()

    def find_periods_position(sent):
        """THIS IS A DOCSTRING"""
        # Tokenize
        #words = tknzr.tokenize(sent)
        words = sent.split()
        periods_pos = []
        symbols_type = []
        rm_periods = [words[0]]
        for i in range(1, len(words)):
            if words[i] in [x for x in SYMBOLS]:
                periods_pos.append(i - len(periods_pos))
                symbols_type.append(words[i])
            else:
                rm_periods.append(words[i])
        return periods_pos, symbols_type, rm_periods

    periods_pos, symbols_type, rm_periods = find_periods_position(sent)
    #print("input:")
    #print " ".join(rm_periods)
    return periods_pos, symbols_type, " ".join(rm_periods)


def sentence_prediction(sent, labelmode):
    """THIS IS A DOCSTRING"""
    def gen_sent(wordlist):
        """THIS IS A DOCSTRING"""
        return "".join(wordlist)
        line = wordlist[0]
        for i in range(1, len(wordlist)):
            if wordlist[i][0] >= 'a' and wordlist[i][0] <= 'z' or \
                wordlist[i][0] >= '0' and wordlist[i][0] <= '9':
                line += " " + wordlist[i]
            else:
                line += wordlist[i]
        return "".join(line)
    words = sent.split()

    def rm_space(line):
        """THIS IS A DOCSTRING"""
        line = re.sub(ur'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', line)
        return line

    endid = 1
    output = [words[0]]
    prob_break = [[] for _ in range(len(words))]# prob_break[t]: prob of break before t
    max_symbol = [","] * len(words)
    last_break_pos = 0
    cur_sent = [words[0]]
    nsent = 0
    history = ""
    while endid < len(words):
        startid = max(endid - FLAGS.max_wait_words, last_break_pos)
        if FLAGS.textb_none == False:
            texta = history + rm_space(" ".join(words[startid: endid]))
            textb = rm_space(" ".join(words[endid: min(endid + FLAGS.appendn, len(words))]))
        else:
            texta = history + gen_sent(words[startid: endid + FLAGS.appendn])
            textb = None
        prob = predict_oneline([texta.encode('utf8'), textb.encode('utf8')])
        prob_break[endid] = 1 - prob[-1]
        sys.stderr.write("TA: " + texta.encode('utf8'))
        if FLAGS.textb_none == False:
            sys.stderr.write("\tTB: " + textb.encode('utf8'))
            #sys.stderr.write("\t".join([str(x) for x in [len(history), len(texta)]]))
        sys.stderr.write("\t[" + " ".join([str(x) for x in prob]) + "]")
        #sys.stderr.write("\t" + str(prob_break[endid]))
        sys.stderr.write("\n")

        #if len(texta.split()) > 50:
        #    prob[1] *= 5
        #    prob[2] *= 2
        max_symbol[endid] = SYMBOLS[prob[0:-1].argmax()]
        detect_pos = endid

        #if endid > startid and prob_break[detect_pos] > 0.6:
        #    break_symbol = max_symbol[detect_pos]
        #    if break_symbol in [",", "|"]:
        #        history = texta
        #        if break_symbol == ",":
        #            history += ","
        #    else:
        #        history = ""

        THRES_LEN = 25
        if endid - startid > 0 and prob_break[detect_pos] > 0.6:
            break_symbol = max_symbol[detect_pos]
            if break_symbol in [",", "|"] or break_symbol == '.' and len(texta.split()) < THRES_LEN:
                history = texta + " "
                if break_symbol in [",", '.']:
                    history += ", "
            else:
                #if break_symbol in ['?', '.']:
                if break_symbol == '?':
                    history = ""
                elif len(texta.split()) >= THRES_LEN:
                    history = texta + " "
                    while len(history.split()) >= THRES_LEN:
                        first_douhao = history.find(",")
                        if first_douhao == -1:
                            break
                        history = history[first_douhao + 1:]
                    if break_symbol in [",", '.']:
                        history += ", "
            output.append(break_symbol)
            cur_sent.append(break_symbol)
            last_break_pos = detect_pos
            print >> f_finalout, " ".join([w.encode('utf8') for w in cur_sent])
            nsent += 1
            cur_sent = []
            if detect_pos < len(words):
                prob_break[detect_pos] = 0
                cur_sent.append(words[detect_pos])
                output.append(words[detect_pos])
        else:
            output.append(words[detect_pos])
            cur_sent.append(words[detect_pos])
            if len(cur_sent) > FLAGS.max_wait_words:
                maxpos = -1
                maxvalue = -1
                for i in range(FLAGS.max_wait_words):
                    if np.mean(prob_break[detect_pos - i]) > maxvalue:
                        maxpos = detect_pos - i
                        maxvalue = np.mean(prob_break[detect_pos - i])
                break_symbol = max_symbol[maxpos]
                if break_symbol in [",", "|"] or break_symbol == '.' and len(texta.split()) < THRES_LEN:
                    zishu = 0
                    while zishu != endid - maxpos:
                        if texta[-1] == " ":
                            texta = texta[:-1]
                            continue
                        while len(texta) and (texta[-1] != " "):
                            texta = texta[:-1]
                        zishu += 1
                        texta = texta[:-1]
                    history = texta + " "
                    #history = texta[:len(texta) - (endid - maxpos)]
                    if break_symbol in [",", '.']:
                        history += ", "
                else:
                    history = ""
                cur_sent = cur_sent[:maxpos - startid]
                output = output[:maxpos - startid + last_break_pos + nsent] + [break_symbol]
                print >> f_finalout, " ".join([w.encode('utf8') for w in cur_sent + [break_symbol]])
                nsent += 1
                endid = maxpos
                last_break_pos = maxpos
                cur_sent = []
                if endid < len(words):
                    prob_break[endid] = 0
                    cur_sent.append(words[endid])
                    output.append(words[endid])
        endid += 1

    if last_break_pos + 2 < len(words):
        output.append("。")
        cur_sent.append("。")
    if len(cur_sent) > 0:
        print >> f_finalout, " ".join(cur_sent)
        nsent += 1

    print("prediction:")
    print(" ".join(output))
    print >> f_nsent, str(nsent)
    return " ".join(output)


def test_dataset(filename): # calculate p/r of a dataset
    """THIS IS A DOCSTRING"""
    ngram = 5

    def fscore(p, r):
        """THIS IS A DOCSTRING"""
        return 2 * p * r / (p + r)
    n_correct_duanAll = 0
    n_correct_duanBiaodian = 0
    n_correct_classifyAll = 0
    n_correct_classifyBiaodian = 0
    n_correct_classifySegment = 0

    n_predict_pos = 0
    n_gt_pos = 0
    n_predict_biaodian = 0
    n_gt_biaodian = 0
    n_predict_segment = 0
    n_gt_segment = 0

    lineid = 0
    latency = []
    onesent = ""
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line != "":
                #onesent = line
                onesent += line + " "
            else:
                onesent = onesent.strip()
                gt_pos, gt_symbols, sent = get_periods_position(onesent)
                predict_sent = sentence_prediction(sent, LABELMODE)
                predict_pos, predict_symbols, _ = get_periods_position(predict_sent)
                for i in range(len(gt_pos)):
                    try:
                        # if the i-th segment in gt_pos can be found in predict_pos
                        j = predict_pos.index(gt_pos[i])
                        n_correct_duanAll += 1
                        if gt_symbols[i] != "|" and predict_symbols[j] != "|": # both predict and gt is biaodian
                            n_correct_duanBiaodian += 1
                        if gt_symbols[i] == predict_symbols[j]:
                            n_correct_classifyAll += 1
                            if gt_symbols[i] != "|": # belongs to biaodian
                                n_correct_classifyBiaodian += 1
                            else:
                                n_correct_classifySegment += 1
                    except: # index j not find
                        pass
                n_predict_pos += len(predict_pos)
                n_gt_pos += len(gt_pos)
                n_predict_biaodian += sum([x != "|" for x in predict_symbols])
                n_gt_biaodian += sum([x != "|" for x in gt_symbols])
                n_predict_segment += sum([x == "|" for x in predict_symbols])
                n_gt_segment += sum([x == "|" for x in gt_symbols])
                onesent = ""
                print("---------Duan All---------")
                print("right:%d\tall_out:%d\tall_ans:%s" \
                        % (n_correct_duanAll, n_predict_pos, n_gt_pos))
                precision = n_correct_duanAll * 1.0 / n_predict_pos
                recall = n_correct_duanAll * 1.0 / n_gt_pos
                if precision * recall == 0:
                    f = 0
                else:
                    f = 2 * precision * recall / (precision + recall)
                print("Precision:%.4f\tRecall:%.4f\tF-score:%.4f" % (precision, recall, f))

                print("---------Duan Biaodian---------")
                print("right:%d\tall_out:%d\tall_ans:%s" \
                        % (n_correct_duanBiaodian, n_predict_biaodian, n_gt_biaodian))
                precision = n_correct_duanBiaodian * 1.0 / n_predict_biaodian
                recall = n_correct_duanBiaodian * 1.0 / n_gt_biaodian
                if precision * recall == 0:
                    f = 0
                else:
                    f = 2 * precision * recall / (precision + recall)
                print("Precision:%.4f\tRecall:%.4f\tF-score:%.4f" % (precision, recall, f))

                print("---------Classify All---------")
                precision = n_correct_classifyAll * 1.0 / n_predict_pos
                recall = n_correct_classifyAll * 1.0 / n_gt_pos
                if precision * recall == 0:
                    f = 0
                else:
                    f = 2 * precision * recall / (precision + recall)
                print("Precision:%.4f\tRecall:%.4f\tF-score:%.4f" % (precision, recall, f))

                print("---------Classify Biaodian---------")
                print("right:%d\tall_out:%d\tall_ans:%s" % \
                        (n_correct_classifyBiaodian, n_predict_biaodian, n_gt_biaodian))
                precision = n_correct_classifyBiaodian * 1.0 / n_predict_biaodian
                recall = n_correct_classifyBiaodian * 1.0 / n_gt_biaodian
                if precision * recall == 0:
                    f = 0
                else:
                    f = 2 * precision * recall / (precision + recall)
                print("Precision:%.4f\tRecall:%.4f\tF-score:%.4f" % (precision, recall, f))

                print("---------Classify Segment---------")
                print("right:%d\tall_out:%d\tall_ans:%s" % \
                        (n_correct_classifySegment, n_predict_segment, n_gt_segment))
                precision = n_correct_classifySegment * 1.0 / n_predict_segment
                recall = n_correct_classifySegment * 1.0 / n_gt_segment
                if precision * recall == 0:
                    f = 0
                else:
                    f = 2 * precision * recall / (precision + recall)
                print("Precision:%.4f\tRecall:%.4f\tF-score:%.4f" % (precision, recall, f))

                for i in range(1, len(predict_pos)):
                    latency.append(predict_pos[i] - predict_pos[i - 1])

                if len(latency) > 0:
                    avglatency = sum(latency) * 1.0 / len(latency)
                    print("Latency: Avg:%.2f\tMax: %d" % (avglatency, max(latency)))
                print("------------------\n")
                onesent = ""

    precision = n_correct_classifySegment * 1.0 / n_predict_pos
    recall = n_correct_classifySegment * 1.0 / n_gt_pos
    print("Precision\tRecall\tF-score")
    print("%.4f\t%.4f\t%.4f" % (precision, recall, f))

f_finalout = open(os.path.join(args.init_checkpoint, "final.out"), "w")
f_nsent = open(os.path.join(args.init_checkpoint, "final.nsent"), "w")

if __name__ == "__main__":
    test_dataset('data/testset/GTC2019/src.transcript.tok')
    f_finalout.close()
    f_nsent.close()
    #predict_oneline(["I have", "a dream"])
