import numpy as np

if __name__ == '__main__':

    filename = 'Amalia_Data/amalia_windrose_full.txt'

    windrose_file = open(filename)

    windrose_data = np.zeros((72, 14))
    i = 0
    for line in windrose_file:
        # print line, i
        windrose_data[i, :] = line.split('\t')
        i += 1
    windrose_file.close()

    windrose_probabilities = np.sum(windrose_data, 1)
    windrose_speeds = np.hstack([np.array([0]), np.arange(4, 17)])
    windrose_directions = np.arange(0, 360, 5)
    print  windrose_directions.shape, windrose_directions

    windrose_speeds_dir_ave = np.zeros(len(windrose_directions))

    for j in range(0, len(windrose_directions)):
        for i in range(0, len(windrose_speeds)):
            windrose_speeds_dir_ave[j] += windrose_speeds[i]*windrose_data[j, i]
        if windrose_probabilities[j] > 0:
            windrose_speeds_dir_ave[j] /= windrose_probabilities[j]

    print 'probabilities', windrose_probabilities.shape, windrose_probabilities
    print 'speeds', windrose_speeds.shape, windrose_speeds
    print 'directionally averaged speeds', windrose_speeds_dir_ave.shape, windrose_speeds_dir_ave

