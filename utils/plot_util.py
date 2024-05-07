import matplotlib.pyplot as plt


def plot_flwr():
    plt.figure(figsize=(12, 8), dpi=100)
    flwr_acc = []
    with open('../log/flower/flwr_fedAvg_10_server_logging', 'r') as f:
        lines = f.readlines()
        for line in lines:
            # print('server-side evaluation' in line)
            if 'server-side evaluation' in line:
                ss = line.split(' ')
                flwr_acc.append(float(ss[-1]))
                print(ss[-1])
    plt.plot(range(len(flwr_acc)), flwr_acc, label='flower')
    plt.xlabel('acc')
    plt.ylabel('round')
    plt.title('server-round-acc plot')
    plt.legend()
    plt.show()


def plot_fedscope():
    plt.figure(figsize=(12, 8), dpi=100)
    fedml_acc = []
    fedscope_acc = []
    # plt.figure(figsize=(12, 8), dpi=100)
    with open('../log/fedscope/fedscope_fedAvg_10_server_logging', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'global eval res' in line:
                ss = line.split(' ')
                print(ss)
                index = ss.index("{'test_acc':")
                print(ss[index + 1])
                s = ss[index + 1]
                idx = s.index(',')
                acc = s[:idx]
                print(acc, float(acc))
                fedscope_acc.append(float(acc))
    print(len(fedscope_acc))
    plt.plot(range(len(fedscope_acc)), fedscope_acc, label='fedscope', color='g')
    plt.xlabel('round')
    plt.ylabel('acc')
    plt.title('server-round-acc plot')

    plt.legend()
    plt.show()


def plot_fedscale():
    plt.figure(figsize=(12, 8), dpi=100)
    fedscale_acc = []
    with open('../log/fedscale/fedscale_mnist_logging', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'FL Testing' in line:
                ss = line.split(' ')
                index = ss.index("'top_1':")
                s = ss[index + 1]
                idx = s.index(',')
                acc = s[:idx]
                print(acc, float(acc))
                fedscale_acc.append(float(acc))
    range_idx = range(0, len(fedscale_acc), 2)

    fedscale_acc = fedscale_acc[0:len(fedscale_acc) + 1:2]
    plt.plot(range_idx, fedscale_acc, label='fedscale', color='r')
    # plt.ylim(0.3, 0.6)
    plt.xlabel('round')
    plt.ylabel('acc')
    plt.title('server-round-acc plot')

    plt.legend()
    plt.show()


def plot_fedml():

    plt.figure(figsize=(12, 8), dpi=100)
    fedml_acc = []
    with open('../log/fedml/fedml_fedAvg_20_cifar10_server_logging', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'test_acc' in line:
                ss = line.split(' ')
                print(ss)
                index = ss.index("{'test_acc':")
                # print(ss[index + 2])
                t = ss[index + 1]
                idx = t.index(',')
                print(t[1:idx], round(float(t[1:idx]), 3))
                fedml_acc.append(round(float(t[1:idx]), 3))
    print(len(fedml_acc))

    plt.plot(range(len(fedml_acc)), fedml_acc, label='fedml', color='y')
    plt.ylim(0, 1)
    # plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=['0', '0.1', "0.2", "0.3", "0.4", "0.5", "0.6"," 0.7"," 0.8"," 0.9", "1"])
    plt.xlabel('round')
    plt.ylabel('acc')
    plt.title('server-round-acc plot')
    plt.legend()
    plt.show()


def plot_all():
    plt.figure(figsize=(12, 8), dpi=100)
    flwr_acc = []
    with open('../log/flower/flwr_fedAvg_10_server_logging', 'r') as f:
        lines = f.readlines()
        for line in lines:
            # print('server-side evaluation' in line)
            if 'server-side evaluation' in line:
                ss = line.split(' ')
                flwr_acc.append(float(ss[-1]))
                print(ss[-1])
    fedscope_acc = []
    # plt.figure(figsize=(12, 8), dpi=100)
    with open('../log/fedscope/fedscope_fedAvg_10_server_logging', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'global eval res' in line:
                ss = line.split(' ')
                index = ss.index("'test_acc':")
                print(ss[index + 1])
                s = ss[index + 1]
                idx = s.index(',')
                acc = s[:idx]
                print(acc, float(acc))
                fedscope_acc.append(float(acc))
    fedml_acc = []
    with open('../log/fedml/fedml_fedAvg_10_server_logging', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'test on server metrics' in line:
                ss = line.split(' ')
                index = ss.index('is')
                # print(ss[index + 2])
                t = ss[index + 1]
                idx = t.index(',')
                print(t[1:idx], round(float(t[1:idx]), 3))
                fedml_acc.append(round(float(t[1:idx]), 3))

    plt.plot(range(len(flwr_acc)), flwr_acc, label='flower')
    plt.plot(range(len(fedscope_acc)), fedscope_acc, label='fedscope')
    plt.plot(range(len(fedml_acc)), fedml_acc, label='fedml')
    plt.xlabel('acc')
    plt.ylabel('round')
    plt.title('server-round-acc plot')
    plt.legend()
    plt.show()


# plot_fedml()
plot_flwr()
# plot_fedscale()
# plot_fedscope()
# plot_all()
# cnt = 0
#
# with open('../log/fedscale/fedscale_mnist_logging', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         if 'FL Testing' in line:
#             ss = line.split(' ')
#             index = ss.index("'top_1':")
#             cnt = cnt + 1
# print(cnt)
