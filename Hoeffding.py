import numpy as np
import matplotlib.pyplot as plt


def main():
    N = 200000
    n = 20
    mat = np.random.binomial(1,0.5, (N,n))

    S = np.zeros(N)
    for i in range(N):
        S[i] = np.sum(mat[i])
        S[i] = S[i]/float(n)
    
    epsilons = np.linspace(0,1, 50)
    emp_prob = np.zeros(50)
    for i in range(50):
        for j in range(N):
            if(abs(S[j] - 0.5) > epsilons[i]):
                emp_prob[i] += 1
        emp_prob[i] /= float(N)
    
    hoef = ((-2)*np.power(epsilons, np.full(50,2)))
    plt.plot(epsilons, emp_prob, 'r')
    plt.plot(epsilons, hoef, 'b')
    plt.show()





if __name__ == '__main__':
    main()