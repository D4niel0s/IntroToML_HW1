import numpy as np
import matplotlib.pyplot as plt


def main():
    N = 200000
    n = 20
    mat = np.random.binomial(1,0.5, (N,n))

    S = np.zeros(N)
    for i in range(N):
        S[i] = np.mean(mat[i])
    
    epsilons = np.linspace(0,1, 50)
    emp_prob = np.zeros(50)
    for i in range(50):
        for j in range(N):
            if(S[j]-0.5>epsilons[i] or 0.5-S[j]>epsilons[i]):
                emp_prob[i] += 1
        emp_prob[i] /= float(N)
    
    hoef = 2*np.exp((-2*n)*np.power(epsilons,2))
    plt.plot(epsilons, hoef, 'b',label='Hoeffding bound')
    plt.plot(epsilons, emp_prob, 'r', label='Empirical probability')
    plt.xlabel("Epsilon")
    plt.ylabel("Probability")
    plt.title("Visualization of the Hoeffding bound")
    plt.legend()
    plt.show()





if __name__ == '__main__':
    main()