from __future__ import division, print_function
from matplotlib.pyplot import figure, legend, plot, show, xlabel, ylabel
from numpy import fromiter, linspace
from random import choice, randint, sample, seed
from sklearn.metrics import adjusted_rand_score


def j(a, b):
    return len(a & b) / len(a | b)


def ari(a, b, p):
    return adjusted_rand_score([(x in a) for x in p], [(x in b) for x in p])


def create_pairing(a, b):
    p = set(x for xs in a for x in xs) | set(x for xs in b for x in xs)

    all_pairings = sorted(
        ((n, m, ari(x, y, p)) for n, x in enumerate(a) for m, y in enumerate(b)),
        key=lambda x: x[-1],
        reverse=True)

    aused = [False] * (len(a) + 1)
    bused = [False] * (len(b) + 1)

    for n, m, s in all_pairings:
        if not aused[n] and not bused[m]:
            aused[n] = bused[m] = True
            yield n, m, s


def similarity(pairing, x):
    return sum(1 for _, _, s in pairing if s > x) / len(pairing)


def adjusted_rand_score_sets(a, b, pairing):
    def f(x, cs):
        for n, c in enumerate(cs):
            if x in c:
                return n
        return -1

    p = set(x for c in a for x in c) | set(x for c in b for x in c)
    ordered_a = [a[x[0]] for x in sorted(pairing, key=lambda x: x[1])]
    return adjusted_rand_score([f(x, ordered_a) for x in p], [f(x, b) for x in p])


def compare_clusters(true_clusters, predicted_clusters_dict):
    figure(facecolor='white')
    xlabel('Minimum Per-Cluster Adjusted RAND Score')
    ylabel('Ratio of Clusters')

    xs = linspace(0, 1, 1000)

    for name, clusters in predicted_clusters_dict.items():
        pairing = list(create_pairing(true_clusters, clusters))
        print('pairing=', [(true_clusters[i], clusters[j], s) for i, j, s in pairing])
        ys = fromiter((similarity(pairing, x) for x in xs), dtype='float')
        label = '{} (ARI={:.3g})'.format(
            name,
            adjusted_rand_score_sets(true_clusters, clusters, pairing))
        plot(xs, ys, label=label)

    legend()
    show()


# Begin Testing ********** ********** ********** ********** ********** ********** ********** **********

def remove_falsy(lst):
    return (x for x in lst if x)


def modify(s, ratio, population):
    t = set(s)
    for x in sample(t, randint(0, int(ratio * len(t)))):
        t.remove(x)
    for x in (choice(population) for _ in range(randint(0, int(ratio * len(t))))):
        t.add(x)
    return t


def plot_modify_vs_adjusted_rand_score(s=0):
    def f():
        seed(s)
        population = list(range(1000)) 
        true_clusters = [set(choice(population) for _ in range(randint(2, 11))) for _ in range(10)]
        l = len(true_clusters)

        for i in range(1, 30):
            x = i / 30
            clusters = [modify(c, x, population) for c in sample(true_clusters, l)]
            clusters = [c for c in clusters if len(c) > 0]
            pairing = list(create_pairing(clusters, true_clusters))
            yield x, adjusted_rand_score_sets(clusters, true_clusters, pairing)

    xs, ys = zip(*f())

    xlabel('Modification')
    ylabel('Adjusted RAND Score')
    plot(xs, ys)
    show()


def compare_test_sets(m1=1/10, m2=3/10, m3=5/10, s=0):
    seed(s)

    population = list(range(100)) 

    true_clusters = [set(choice(population) for _ in range(randint(2, 11))) for _ in range(20)]
    l = len(true_clusters)

    clusters1 = list(remove_falsy(modify(c, m1, population) for c in sample(true_clusters, l)))
    clusters2 = list(remove_falsy(modify(c, m2, population) for c in sample(true_clusters, l)))
    clusters3 = list(remove_falsy(modify(c, m3, population) for c in sample(true_clusters, l)))

    compare_clusters(true_clusters, {
        'Theia': clusters1,
        'SNMNMF': clusters2,
        'Algorithm3': clusters3
    })


# Uncomment these to see the testing results
# compare_test_sets()
# plot_modify_vs_adjusted_rand_score()

# End Testing ********** ********** ********** ********** ********** ********** ********** **********

if __name__ == '__main__':
    compare_test_sets()
