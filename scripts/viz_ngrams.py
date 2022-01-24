import argparse
from testperanto.analysis import plot_statistic, singleton_proportion, type_count_over_time
from testperanto.analysis import powers_of_2, multiples_of_1000
from testperanto.corpora import stream_lines


def main(corpora, metric_name):
    if metric_name == "types":
        metric = type_count_over_time
        axes = "loglog"
    else:
        metric = singleton_proportion
        axes = "semilogx"
    streams = [stream_lines(filename) for filename in corpora]
    plot_statistic(metric, streams, powers_of_2, axes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the behavior of ngram streams.')
    parser.add_argument('-c', '--corpora', nargs='+', required=True,
                        help='names of the corpus files')
    parser.add_argument('-m', '--metric', type=str,
                        help='metric to plot (e.g. "types", "sprop")')
    args = parser.parse_args()
    main(args.corpora, args.metric)
