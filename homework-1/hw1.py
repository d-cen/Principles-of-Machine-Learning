import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mu = 0
sigma = 1
samples = 1000



def plotPDF():
    x = np.linspace(-3, 3, 1000)
    y = stats.norm.pdf(x, mu, sigma)
    #y = (1 / np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2)
    plt.plot(x, y, label='Standard Normal Distribution', color='coral')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

def histPDF():
    X = np.random.normal(mu, sigma, samples)
    (mu_, sigma_) = stats.norm.fit(X)
    x = np.linspace(min(X), max(X), 1000)
    curve = stats.norm.pdf(x, mu_, sigma_)
    print("Mean:    ", mu_, "\nStd dev: ", sigma_)
    plt.figure(1)
    plt.hist(X, bins=4, density=True)
    plt.plot(x, curve)
    plt.figure(2)
    plt.hist(X, bins=1000, density=True)
    plt.plot(x, curve)

def optimalBin():
    X = np.random.normal(mu, sigma, samples)
    J_h = []
    num_bins = range(1, 200+1)
    for m in num_bins:
        (v, bins, patches) = plt.hist(X, bins=m)
        plt.clf()
        h = bins[1] - bins[0]
        p = v / samples
        p_sum = sum(p ** 2)
        J = (2 / (h * (samples - 1)) - (samples + 1) * (p_sum / (h * (samples - 1))))
        J_h.append(J)
    
    # Find m that minimizes J_h
    opt_bins = num_bins[np.argmin(J_h)]
    
    # Fit normalized Gaussian curve to data
    mu_, sigma_ = stats.norm.fit(X)
    x = np.linspace(min(X), max(X), samples)
    curve = stats.norm.pdf(x, mu_, sigma_)
    
    plt.figure(1)
    plt.plot(num_bins, J_h)
    
    # Normalized histogram with fitted Gaussian curve
    plt.figure(2)
    plt.hist(X, bins=opt_bins, density=True)
    plt.plot(x, curve)

def contourPDF():
    x1 = np.linspace(-1, 5, 100)
    y1 = np.linspace(0, 10, 100)
    (x, y) = np.meshgrid(x1, y1)
    position = np.empty(x.shape + (2,))
    position[:, :, 0] = x
    position[:, :, 1] = y
    z = stats.multivariate_normal([2., 6.], [[2., 1.], [1., 2.]])
    z = z.pdf(position)
    plt.contour(x, y, z)
    
def scatterPDF():
    m = np.array([0, 0])
    c = np.array([[1, 0], [0, 1]])
    (x, y) = np.random.multivariate_normal(mean=m, cov=c, size=5000).T
    plt.scatter(x, y, marker='.')
    
    e_val, e_vec = np.linalg.eig([[2, 1], [1, 2]])
    l = [[np.sqrt(3), 0], [0, 1]]
    matrix = np.dot(np.dot(e_vec, l), e_vec.T)
    stack = np.column_stack((x, y))
    x_t, y_t = x, y
    for i in range(5000):
        (x_t[i], y_t[i]) = np.dot(matrix, stack[i]) + np.array([2, 6])
    plt.scatter(x_t, y_t, marker='.', color='coral')
    
    

if __name__ == "__main__":
    pass