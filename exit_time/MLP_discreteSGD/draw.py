import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import numpy as np
from matplotlib.ticker import StrMethodFormatter
def draw(sharpness_results, lr_results, batch_size_results, r_results):
    font = {'size' : 29}
    plt.rcParams.update({
      "text.usetex": True,
      "font.family": "Times"
    })
    plt.rcParams["font.family"] = "Times"
    matplotlib.rc('font', **font)
    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    plt.rcParams['axes.titley'] = 1.05 
    # Sharpness 
    (x, y, std, log_std, _) = sharpness_results
    x_1 = x
    y_1 = np.log(y)**2
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x_1))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    plt.xlabel("$\lambda_\mathrm{max}$: Sharpness")
    plt.ylabel("$\log(\mathbf{E}[\\nu])$")
    plt.errorbar(x_1, y_1, yerr=log_std+np.sqrt(2)*np.sqrt(y_1)*log_std, fmt='.', capsize=2) 
    plt.plot(x_1, m_1*x_1 + c_1)
    plt.legend([f'Linear Correlation: {coeff_1:.3g}'])
    plt.title('$\mathbf{E}[\\nu]\sim \exp(\lambda_\mathrm{max}^{-1})$')
    plt.tight_layout()
    plt.savefig("discrete_SGD_sharpness.pdf", dpi=100)
    # Learning rate
    plt.clf()
    (x, y, std, log_std, _) = lr_results
    x_1 = x
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    plt.xlabel("$\eta$: Learning rate")
    plt.ylabel("$\log(\mathbf{E}[\\nu])$")
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.1e}'))
    plt.locator_params(axis="x", nbins=5)
    plt.errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    plt.plot(x_1, m_1*x_1 + c_1)
    plt.legend([f'Linear Correlation: {coeff_1:.3g}'])
    plt.title('$\mathbf{E}[\\nu]\sim \exp(\eta^{-1})$')
    plt.tight_layout()
    plt.savefig("discrete_SGD_learning_rate.pdf", dpi=100)
    # Batch size
    plt.clf()
    (x, y, std, log_std, _) = batch_size_results
    x_1 = x
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    plt.xlabel("$B$: Batch size")
    plt.ylabel("$\log(\mathbf{E}[\\nu])$")
    plt.errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    plt.plot(x_1, m_1*x_1 + c_1)
    plt.legend([f'Linear Correlation: {coeff_1:.3g}'])
    plt.title('$\mathbf{E}[\\nu] \sim \exp(B)$')
    plt.tight_layout()
    plt.savefig("discrete_SGD_batch_size.pdf", dpi=100)
    # R
    plt.clf()
    (x, y, std, log_std, _) = r_results
    x_1 = x**2
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    plt.xlabel("$\Delta L$: Depth")
    plt.ylabel("$\log(\mathbf{E}[\\nu])$")
    plt.errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    plt.plot(x_1, m_1*x_1 + c_1)
    plt.legend([f'Linear Correlation: {coeff_1:.3g}'])
    plt.title('$\mathbf{E}[\\nu] \sim \exp(\Delta L)$')
    plt.tight_layout()
    plt.savefig("discrete_SGD_delta_L.pdf", dpi=100)

    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 12))
    # Sharpness 
    (x, y, std, log_std, _) = sharpness_results
    x_1 = x
    y_1 = np.log(y)**2
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x_1))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    ax1[0].set_xlabel("$\lambda_\mathrm{max}$: Sharpness")
    ax1[0].set_ylabel("$\log(\mathbf{E}[\\nu])$")
    ax1[0].errorbar(x_1, y_1, yerr=log_std+np.sqrt(2)*np.sqrt(y_1)*log_std, fmt='.', capsize=2) 
    ax1[0].plot(x_1, m_1*x_1 + c_1)
    ax1[0].legend([f'Linear Correlation: {coeff_1:.3g}'])
    ax1[0].set_title('$\mathbf{E}[\\nu]\sim \exp(\lambda_\mathrm{max}^{-1/2})$')
    # Learning rate
    (x, y, std, log_std, _) = lr_results
    x_1 = x
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    ax1[1].set_xlabel("$\eta$: Learning rate")
    ax1[1].set_ylabel("$\log(\mathbf{E}[\\nu])$")
    ax1[1].errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    ax1[1].plot(x_1, m_1*x_1 + c_1)
    ax1[1].legend([f'Linear Correlation: {coeff_1:.3g}'])
    ax1[1].set_title('$\mathbf{E}[\\nu]\sim \exp(\eta^{-1})$')
    # Batch size
    (x, y, std, log_std, _) = batch_size_results
    x_1 = x
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    ax2[0].set_xlabel("$B$: Batch size")
    ax2[0].set_ylabel("$\log(\mathbf{E}[\\nu])$")
    ax2[0].errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    ax2[0].plot(x_1, m_1*x_1 + c_1)
    ax2[0].legend([f'Linear Correlation: {coeff_1:.3g}'])
    ax2[0].set_title('$\mathbf{E}[\\nu] \sim = \exp(B)$')
    # R
    (x, y, std, log_std, _) = r_results
    x_1 = x**2
    y_1 = np.log(y)
    coeff_1, _ = stats.pearsonr(x_1, y_1)
    A = np.vstack([x_1, np.ones(len(x))]).T
    m_1, c_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]
    ax2[1].set_xlabel("$\Delta L$: Depth")
    ax2[1].set_ylabel("$\log(\mathbf{E}[\\nu])$")
    ax2[1].errorbar(x_1, y_1, yerr=log_std, fmt='.', capsize=2) 
    ax2[1].plot(x_1, m_1*x_1 + c_1)
    ax2[1].legend([f'Linear Correlation: {coeff_1:.3g}'])
    ax2[1].set_title('$\mathbf{E}[\\nu] \sim \exp(\Delta L)$')
    plt.tight_layout()
    fig.savefig("results.png",dpi=500)

if __name__=='__main__':
    sharpness_results  = np.load(f"./sharpness_results.npy")
    lr_results         = np.load(f"./lr_results.npy")
    batch_size_results = np.load(f"./batch_size_results.npy")
    r_results          = np.load(f"./r_results.npy")
    draw(sharpness_results, lr_results, batch_size_results, r_results)
