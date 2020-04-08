# coding=utf-8

import reader.task_reader as task_reader
from model.ernie import ErnieConfig
from finetune.classifier import create_model_predict, evaluate
from optimization import optimization
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint
import paddle.fluid as fluid
#from finetune_args import parser
import datetime, csv
import os, re
import numpy as np
import sys
from datetime import datetime
import pdb

from flask import Flask, request, Response
import json

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'

place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
exe = fluid.Executor(place)

startup_prog = fluid.Program()
#args = parser.parse_args()
class args:
    ernie_config_path="data/ERNIE_Large_en_stable-2.0.0/ernie_config.json"
    use_cuda=True
    do_train=False
    do_val=False
    do_test=True
    batch_size=32
    test_set="data/chnsenticorp/test.tsv"
    vocab_path="data/ERNIE_Large_en_stable-2.0.0/vocab.txt"
    init_checkpoint="checkpoints_cls2_LEN40_APPEND5_SEGMENT_force_context/step_12000" # 8808

    max_seq_len=256
    num_labels=5
    use_fp16=False
    label_map_config=None
    do_lower_case=True
    in_tokens=False
    random_seed=1

ernie_config = ErnieConfig(args.ernie_config_path)
ernie_config.print_config()
test_prog = fluid.Program()

with fluid.program_guard(test_prog, startup_prog):
    with fluid.unique_name.guard():
        _, graph_vars = create_model_predict(
            args,
            ernie_config=ernie_config,
            is_prediction=True)
LABELMODE = [str(x) for x in range(args.num_labels - 1)]

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

#print(datetime.datetime.now())

def rm_space(line):
    line = re.sub(ur'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', line)
    return line

class Example:
    def __init__(self, line):
        self._fields = ["text_a", "label"]
        self.text_a = line
        self.label = 1
    def __init__(self, texta, textb):
        self._fields = ['text_a', 'text_b', 'label']
        self.text_a = rm_space(texta)
        self.text_b = rm_space(textb)
        self.label = 1

class Sample:
    def __init__(self, values):
        self.names = ['eval_placeholder_0', 'eval_placeholder_1', 'eval_placeholder_2', 'eval_placeholder_4', 'eval_placeholder_3']
        self.values = values
        #self.values = [[val] for val in values]
    def gen(self):
        sample = dict(zip(self.names, self.values[0:5]))
        return sample

def predict_oneline(line):
    texta, textb = line
    if textb == None:
        example = Example(texta)
    else:
        dt_string = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print >> sys.stderr, dt_string
        print >> sys.stderr, "texta:", texta
        print >> sys.stderr, "textb:", textb
        example = Example(texta, textb)
    data_gen = reader._prepare_one_sample(example, 1)
    sample = Sample(data_gen)
    sample = sample.gen()

    #print("Final test result:")
    fetch_list = [graph_vars["probs"].name]

    np_probs = exe.run(program=test_prog, feed = sample, fetch_list=fetch_list, use_program_cache=True)
    np_probs = np_probs[0][0]
    #if texta.startswith(u'\u6211\u4eec') and len(texta) < 5:
    #    print(texta.encode('utf8'))
    #    np_probs[0:-1] = 0
    #    np_probs[-1] = 1.0
    #print(datetime.datetime.now())
    #print(np_probs)
    print >> sys.stderr, np_probs
    dt_string = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print >> sys.stderr, dt_string
    return np_probs

@app.route('/prob_paddle', methods = ['POST'])
def prob():
    #print request
    #print request.json
    line = request.json['line']
    probability = predict_oneline(line)
    res = {'prob': probability.tolist()}
    return Response(json.dumps(res), mimetype = 'application/json')


class FLAGS:
    max_wait_words = 40
    waitn = 3
    textb_none = True

def get_periods_position(sent):
    sent = sent.strip().lower().decode('utf8')
    sent = re.sub(r", |\"|- |:","",sent)
    sent = re.sub("ph.d.", "phd", sent)
    #tknzr = TweetTokenizer()

    def find_periods_position(sent):
        # Tokenize
        #words = tknzr.tokenize(sent)
        words = sent.split()
        periods_pos = []
        rm_periods = [words[0]]
        for i in range(1, len(words)):
            if words[i] == "|":
                periods_pos.append(i - len(periods_pos))
            else:
                rm_periods.append(words[i])
        return periods_pos, rm_periods

    periods_pos, rm_periods = find_periods_position(sent)
    #print("input:")
    #print " ".join(rm_periods)
    return periods_pos, " ".join(rm_periods)

def sentence_prediction(sent, labelmode):
    def gen_sent(wordlist):
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
    endid = 1
    output = [words[0]]
    prob_break = [[] for _ in range(len(words))]# prob_break[t]: prob of break before t
    last_break_pos = 0
    cur_sent = [words[0]]
    nsent = 0
    while endid < len(words):
        startid = max(endid - FLAGS.max_wait_words, last_break_pos)
        if FLAGS.textb_none == False:
            texta = "".join(words[startid : endid])
            textb = "".join(words[endid : min(endid+FLAGS.waitn, len(words))])
        else:
            texta = gen_sent(words[startid : endid + FLAGS.waitn])
            textb = None
        prob = predict_oneline(texta.encode('utf8'), textb.encode('utf8'))
        for i in range(FLAGS.waitn + 1):
            if endid - i + FLAGS.waitn > 0 and endid - i + FLAGS.waitn < len(words):
                prob_break[endid + FLAGS.waitn - i].append(prob[i])
        detect_pos = endid
        if detect_pos > 0 and np.mean(prob_break[detect_pos]) > 0.4:
            break_symbol = "|"
            output.append(break_symbol)
            cur_sent.append(break_symbol)
            last_break_pos = detect_pos
            print >> f_finalout, " ".join([w.encode('utf8') for w in cur_sent])
            nsent += 1
            cur_sent = []
            cur_sent.append(words[detect_pos])
            output.append(words[detect_pos])
            for i in range(FLAGS.waitn + 1):
                if detect_pos + i < len(words):
                    prob_break[detect_pos + i] = []
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
                cur_sent = cur_sent[:maxpos - (detect_pos - FLAGS.max_wait_words)]
                output = output[:maxpos - (detect_pos - FLAGS.max_wait_words) + last_break_pos + nsent ] + ["|"] 
                # NOTE: no break_symbol added here
                print >> f_finalout, " ".join([w.encode('utf8') for w in cur_sent] + ["|"])
                nsent += 1
                endid = maxpos
                last_break_pos = maxpos
                cur_sent = []
                cur_sent.append(words[endid])
                output.append(words[endid])
                for i in range(FLAGS.waitn + 1):
                    if detect_pos + i < len(words):
                        prob_break[endid + i] = []
        endid += 1

    if last_break_pos + 2 < len(words):
        output.append("|")
        cur_sent.append("|")
    if len(cur_sent) > 0:
        print >> f_finalout, " ".join([w.encode('utf8') for w in cur_sent])
        nsent += 1

    print("prediction:")
    print(" ".join(output).encode('utf8'))
    print >> f_nsent, str(nsent)
    return " ".join(output).encode('utf8')

def test_dataset(filename): # calculate p/r of a dataset
    ngram = 5
    def fscore(p,r):
        return 2*p*r/(p+r)
    n_correct_prediction = 0
    n_predict_pos = 0
    n_gt_pos = 0
    lineid = 0
    onesent = ""
    latency = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                onesent += line + " | "
    onesent = onesent.strip()
    gt_pos, sent = get_periods_position(onesent)
    predict_sent = sentence_prediction(sent, LABELMODE)
    predict_pos, _ = get_periods_position(predict_sent)
    for pos in gt_pos:
        if pos in predict_pos:
            n_correct_prediction += 1
    n_predict_pos += len(predict_pos)
    n_gt_pos += len(gt_pos)
    onesent = ""
    precision = n_correct_prediction * 1.0 / n_predict_pos
    recall = n_correct_prediction * 1.0 / n_gt_pos
    if precision * recall == 0:
        f = 0
    else:
        f = 2 * precision * recall / (precision + recall)
    print("------------------")
    print("Precision:%.2f\tRecall:%.2f\tF-score:%.2f" % (precision, recall, f))

    for i in range(1, len(predict_pos)):
        latency.append(predict_pos[i] - predict_pos[i-1])

    if len(latency) > 0:
        avglatency = sum(latency) * 1.0 / len(latency)
        print("Latency: Avg:%.2f\tMax: %d" % (avglatency, max(latency)))
    print("------------------\n")

    precision = n_correct_prediction * 1.0 / n_predict_pos
    recall = n_correct_prediction * 1.0 / n_gt_pos
    print("Precision\tRecall\tF-score")
    print("%.2f\t%.2f\t%.2f" % (precision, recall, f))

f_finalout = open(os.path.join(args.init_checkpoint, "final.out"), "a")
f_nsent= open(os.path.join(args.init_checkpoint, "final.nsent"), "a")

if __name__ == "__main__":
  #test_dataset('data/source/test.zh')
  app.run(host = '10.255.124.15', port = 8811)
  f_finalout.close()
  f_nsent.close()
  #predict_oneline("在下载完模型和任务数据后运行")

