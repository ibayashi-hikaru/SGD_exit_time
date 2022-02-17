
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import numpy as np
def draw(sharpness_results, lr_results, batch_size_results, r_results):
    font = {'size' : 15}
    matplotlib.rc('font', **font)
    # Sharpness 
    (x, y, std, log_std, _) = sharpness_results
    x_1 = x
    y_1 = y
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x_1))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    plt.xlabel("$\lambda$:sharpness")
    plt.ylabel("$\mathbf{E}[\\tau]$")
    plt.errorbar(x_1, y_1, yerr=std, fmt='.', capsize=2) 
    plt.plot(x_1, m_1*x_1 + c_1)
    plt.legend([f'Linear Correlation: {coeff_1:.3g}'])
    plt.title('$\mathbf{E}[\\tau] \perp \!\!\!\! \perp \lambda$')
    plt.tight_layout()
    plt.show()
    plt.savefig("sharpness.png", dpi=100)
    #
    # Learning rate
    # (x, y, std, log_std, _) = lr_results
    # x_1 = x
    # y_1 = np.log(y)
    # coeff_1, _ = stats.pearsonr(x_1, y_1)
    # A = np.vstack([x_1, np.ones(len(x))]).T
    # m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    # plt.set_xlabel("$\eta$: Learning rate")
    # plt.set_ylabel("$\log(\mathbf{E}[\\tau])$")
    # plt.errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    # plt.plot(x_1, m_1*x_1 + c_1)
    # plt.legend([f'Linear Correlation: {coeff_1:.3g}'])
    # plt.set_title('$\mathbf{E}[\\tau]\sim \exp(\eta^{-1})$')
    # # Batch size
    # (x, y, std, log_std, _) = batch_size_results
    # x_1 = x
    # y_1 = np.log(y)
    # coeff_1, _ = stats.pearsonr(x_1, y_1)
    # A = np.vstack([x_1, np.ones(len(x))]).T
    # m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    # plt.set_xlabel("$B$: Batch size")
    # plt.set_ylabel("$\log(\mathbf{E}[\\tau])$")
    # plt.errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    # plt.plot(x_1, m_1*x_1 + c_1)
    # plt.legend([f'Linear Correlation: {coeff_1:.3g}'])
    # plt.set_title('$\mathbf{E}[\\tau] \sim = \exp(B)$')
    # # R
    # (x, y, std, log_std, _) = r_results
    # x_1 = x**2
    # y_1 = np.log(y)
    # coeff_1, _ = stats.pearsonr(x_1, y_1)
    # A = np.vstack([x_1, np.ones(len(x))]).T
    # m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    # plt.set_xlabel("$\Delta L$: depth of minimum")
    # plt.set_ylabel("$\log(\mathbf{E}[\\tau])$")
    # plt.errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    # plt.plot(x_1, m_1*x_1 + c_1)
    # plt.legend([f'Linear Correlation: {coeff_1:.3g}'])
    # plt.set_title('$\mathbf{E}[\\tau] \sim \exp(\Delta L)$')

    #
    #
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 12))
    # Sharpness 
    (x, y, std, log_std, _) = sharpness_results
    x_1 = x
    y_1 = y
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x_1))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    ax1[0].set_xlabel("$\lambda$:sharpness")
    ax1[0].set_ylabel("$\mathbf{E}[\\tau]$")
    ax1[0].errorbar(x_1, y_1, yerr=std, fmt='.', capsize=2) 
    ax1[0].plot(x_1, m_1*x_1 + c_1)
    ax1[0].legend([f'Linear Correlation: {coeff_1:.3g}'])
    ax1[0].set_title('$\mathbf{E}[\\tau] \perp \!\!\!\! \perp \lambda$')
    #
    # Learning rate
    (x, y, std, log_std, _) = lr_results
    x_1 = x
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    ax1[1].set_xlabel("$\eta$: Learning rate")
    ax1[1].set_ylabel("$\log(\mathbf{E}[\\tau])$")
    ax1[1].errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    ax1[1].plot(x_1, m_1*x_1 + c_1)
    ax1[1].legend([f'Linear Correlation: {coeff_1:.3g}'])
    ax1[1].set_title('$\mathbf{E}[\\tau]\sim \exp(\eta^{-1})$')
    # Batch size
    (x, y, std, log_std, _) = batch_size_results
    x_1 = x
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    ax2[0].set_xlabel("$B$: Batch size")
    ax2[0].set_ylabel("$\log(\mathbf{E}[\\tau])$")
    ax2[0].errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    ax2[0].plot(x_1, m_1*x_1 + c_1)
    ax2[0].legend([f'Linear Correlation: {coeff_1:.3g}'])
    ax2[0].set_title('$\mathbf{E}[\\tau] \sim = \exp(B)$')
    # R
    (x, y, std, log_std, _) = r_results
    x_1 = x**2
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    ax2[1].set_xlabel("$\Delta L$: depth of minimum")
    ax2[1].set_ylabel("$\log(\mathbf{E}[\\tau])$")
    ax2[1].errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    ax2[1].plot(x_1, m_1*x_1 + c_1)
    ax2[1].legend([f'Linear Correlation: {coeff_1:.3g}'])
    ax2[1].set_title('$\mathbf{E}[\\tau] \sim \exp(\Delta L)$')

    plt.tight_layout()
    plt.show()
    fig.savefig("results.png", dpi=500)

if __name__=='__main__':
    sharpness_results  = np.load(f"./sharpness_results.npy")
    lr_results         = np.load(f"./lr_results.npy")
    batch_size_results = np.load(f"./batch_size_results.npy")
    r_results          = np.load(f"./r_results.npy")
    draw(sharpness_results, lr_results, batch_size_results, r_results)
