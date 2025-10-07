"""
Goal
---
1. Read test results from log.txt.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt.txt
    seed2/
        log.txt.txt
    seed3/
        log.txt.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt.txt
            ...
        seed2/
            log.txt.txt
            ...
        seed3/
            log.txt.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict
import os

from dassl.utils import check_isfile, listdir_nohidden
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator
import pickle


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    # print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        # print(msg)

    output_results = OrderedDict()

    # print("===")
    # print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        # print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
        output_results['std'] = std
    # print("===")

    return output_results


def main(args, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            print(f"* {key}: {avg:.2f}%")


    else:
        z = parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )

    return z

if __name__ == "__main__":
    case = 'btn'  #final, few, btn
    data = 'ImageNet' #[ OxfordPets  Caltech101 FGVCAircraft StanfordCars DescribableTextures Food101
                                # SUN397  UCF101 OxfordFlowers EuroSAT]

    Method = ["Uniform"]          #Forgetting  ,"Uncertainty", "Herding", "Submodular", "Glister", "GraNd", "Craig", "Cal"
    if case == 'final':
        scope = [0.05,0.1,0.2,0.3,0.5,1.0]   #0.05, 0.1, 0.2, 0.3, 0.5, 1.0
        few_scope = [16]  #[1, 2, 4, 8, 16]
    elif case == 'few':
        scope = [1.0]   #0.05, 0.1, 0.2, 0.3, 0.5, 1.0
        few_scope = [1, 2, 4, 8, 16]  #[1, 2, 4, 8, 16]
    elif case == 'btn':
        scope = [1.0]   #0.05, 0.1, 0.2, 0.3, 0.5, 1.0
        few_scope = [16]  #[1, 2, 4, 8, 16]
    # 'Forgetting', 'Herding', 'Submodular' (GraphCut/Facility Location), 'Glister',
    # 'GraNd', 'Craig', 'Cal']
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str,default='output', help="path to directory")
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )
    args = parser.parse_args()

    end_signal = "Finish training"
    if args.test_log:
        end_signal = "=> result"

    pallete = [
        '95a2ff',
        'fa8080',
        'ffc076',
        'bf19ff',
        '87e885',
        'f9e264',
        'bdb76b',
        '5f45ff',
        'cb9bff',
        '009db2',
        '0090ff',
        '314976',
        '765005',
    ]
    res_l = []
    plt.figure(figsize=(8,8))  #30,15
    plt.style.use('seaborn-darkgrid')
    ax = plt.subplot()
    for num_m,m in enumerate(Method):
        if m != 'Uniform':
            s = scope[:-1]
        else:
            s = scope
        for i in range(len(s)):
            for few in range(len(few_scope)):
                args.directory = os.path.join('output_'+case,data,m + '_'+ str(s[i]) + '_'+str(few_scope[few]))
                res = main(args, end_signal)
                if case == 'final':
                    args.directory = os.path.join('/home/ubuntu/VLTuning/multimodal-prompt-learning-main/output',data,m + '_'+ str(s[i]))
                    res_base = main(args, end_signal)
                    print(f"{data}_Baseline_{s[i]}/{data}_Updated_{s[i]}: {res_base['accuracy']:.2f}+{res_base['std']:.2f} / {res['accuracy']:.2f}+{res['std']:.2f}")
                elif case == 'btn':
                    args.directory = os.path.join('output_'+case + '_test',data,m + '_'+ str(s[i]) + '_'+str(few_scope[few]))
                    res_base = main(args, 'Evaluate on the *test* set')
                    harmony = 2 / (1/res_base['accuracy'] + 1/res['accuracy'])
                    print(f"{data}_Base_{s[i]}/{data}_Novel_{s[i]}/HM: {res['accuracy']:.2f} / {res_base['accuracy']:.2f} /  {harmony:.2f}")
                else:
                    print(f"{data}_Baseline_{s[i]}_shot_{few_scope[few]}: {res['accuracy']:.2f}+{res['std']:.2f}")


                if m == 'Uniform' and i == (len(s)-1):
                    final = res['accuracy']
                res_l.append(res['accuracy'])
            if m != 'Uniform':
                res_l.append(final)

            x = range(0, len(scope))
            # if m != 'Uniform':
            #     ax.plot(x, res_l, label=Method[num_m], linewidth=2, linestyle='-',color='#'+pallete[num_m], marker='o',markersize=6)
            # else:
            #     ax.plot(x, res_l, label=Method[num_m], linewidth=5, linestyle='--',color='#'+pallete[num_m], marker='o',markersize=10)

        res_l = []

    ax.grid(True)
    ax.set_xlabel('Data Volume', fontsize=20, fontdict={'family': 'Times New Roman', 'weight': "medium"})
    ax.set_ylabel('Accuracy (%)', fontsize=20, fontdict={'family': 'Times New Roman', 'weight': "medium"})
    ax.legend(prop={'size': 18, 'family': 'Times New Roman', 'weight': "bold"}, frameon=True)

    plt.title(data, fontdict={'family': 'Times New Roman', 'weight': "semibold"}, fontsize=20)
    plt.xticks(range(len(scope)),scope,fontsize=15)
    plt.yticks(fontsize=15)
    # plt.ylim((65,79))

    # save_name = 'Basline_fig2'
    # if not os.path.exists(save_name):
    #     os.mkdir(save_name)
    # plt.savefig(os.path.join(save_name,data+'.pdf'),bbox_inches='tight')
    # plt.show()
