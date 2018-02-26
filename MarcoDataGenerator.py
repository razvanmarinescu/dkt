__author__ = 'mlorenzi'
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator(object):
    def __init__(self, Nbiom, interval, param, Nsubs):
        self.Nbiom = Nbiom
        self.YData = []
        self.YDataNoNoise = []
        self.XData = []
        self.model = []
        self.Nsubs = Nsubs
        self.interval = interval
        self.param = param
        self.shiftData = []

        np.random.seed(1)

        for i in range(Nbiom):
            shift = np.random.randint(interval[0], interval[1])
            self.shiftData.append([shift + l for l in range(np.int(np.float(interval[0])/2), np.int(np.float(interval[1])/2))])
            self.model.append(self.f(self.shiftData[i], param[i][0], param[i][1]))
            #self.model.append(self.f(range(interval[0], interval[1]), param[i][0], param[i][1]))

        self.XData.append([])
        for k in range(self.Nsubs):
            start = np.random.randint(interval[1] - 5)
            Nel = np.random.randint(1, 5)
            # sequence = np.sort(random.sample(range(5), Nel))
            sequence = np.sort(np.random.choice(range(5), Nel, replace=False))
            Xd = [l + start for l in sequence]
            self.XData[0].append(Xd)

        for i in range(Nbiom-1):
            self.XData.append([])
            self.XData[i+1] = self.XData[0]

        for i in range(Nbiom):
            self.YData.append([])
            self.YDataNoNoise.append([])
            for l in range(self.Nsubs):
                obs = np.array(self.model[i])[self.XData[i][l]] + param[i][2] * np.random.randn(len(self.XData[i][l]))
                self.YData[i].append(obs)
                self.YDataNoNoise[i].append(np.array(self.model[i])[self.XData[i][l]] )

        self.ZeroXData = []
        for i in range(self.Nbiom):
            self.ZeroXData.append([])
            for k in range(self.Nsubs):
                self.ZeroXData[i].append(np.array(self.XData[i][k]) - np.float (self.XData[i][k][len(self.XData[i][k])-1]+self.XData[i][k][0])/2)


    def f(self, X, L, k):
        return [L / (1 + np.exp(-k * i)) for i in X]

    def predPop(self, X, b):
      # return self.f(X, self.param[b][0], self.param[b][1])
      return np.array(self.model[b])[X]


    def plot(self, mode):
        datamin = 0
        datamax = 1

        if mode == 0:
            elements = [el for sublist in self.ZeroXData for item in sublist for el in item]
            datamin = np.min(elements)
            datamax = np.max(elements)
        if mode == 1:
            elements = [el for sublist in self.XData for item in sublist for el in item]
            datamin = np.min([elements])
            datamax = np.max([elements])

        datarange = [el for sublist in self.YData for item in sublist for el in item]
        axes = plt.gca()
        axes.set_xlim([datamin, datamax])
        axes.set_ylim([np.min(datarange) - 0.5, np.max(datarange) + 0.5])
        fig = plt.figure()
        Blues = plt.get_cmap('prism')

        if mode == 0:
            for i in np.arange(self.Nbiom):
                for k in np.arange(self.Nsubs):
                    plt.plot(self.ZeroXData[i][k], self.YData[i][k], color=Blues(i * 2), linewidth=1, linestyle='--')
          #  fig.savefig('/Users/mlorenzi/Desktop/ipmc/mode0.png')
        if mode == 1:
            for i in np.arange(self.Nbiom):
                print(self.shiftData[i])
                plt.plot( range(len(self.model[i])),self.model[i],  color=Blues(i * 2), linewidth=3)
                plt.annotate("biom.  " + str(i), xy = (len(self.model[i]),self.model[i][len(self.model[i])-1]), color=Blues(i * 2), textcoords='data')
                for k in np.arange(self.Nsubs):
                    plt.plot(self.XData[i][k], self.YData[i][k], color=Blues(i * 2),  linewidth=1, linestyle='--')
         #   fig.savefig('/Users/mlorenzi/Desktop/ipmc/mode1.png')
        fig.show()


    def OutputTimeShift(self):
        time_shift = np.zeros(len(self.XData[0]))
        for l in range(len(self.XData[0])):
            time_shift[l] = self.XData[0][l][0]

        return time_shift