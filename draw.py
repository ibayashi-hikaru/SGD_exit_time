
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
def draw(config_fn, sharpness_results, lr_results, batch_size_results, r_results):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 3, figsize=(12, 16))
    plt.suptitle(config_fn, x=0.05, y=1)
    # Sharpness 
    (x, y, std, log_std, _) = sharpness_results
    y += 0.00001
    # coeff, _ = stats.pearsonr(x, y)
    # A = np.vstack([x, np.ones(len(x))]).T
    # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # #
    # ax1[0].set_xlabel("sharpness")
    # ax1[0].set_ylabel("exit time")
    # ax1[0].errorbar(x, y, yerr=std, fmt='.k') 
    # ax1[0].plot(x, m*x + c) 
    # # ax1[0].set_ylim(bottom=0, top=None)
    # ax1[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    #
    x_2 = x
    y_2 = np.log(y)**2
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x_2))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax1[1].set_xlabel("$\lambda$")
    ax1[1].set_ylabel("$\left(\log(\mathbf{E}[\\tau])\\right)^2$")
    ax1[1].errorbar(x_2, y_2, yerr=log_std**2, fmt='.k') 
    ax1[1].plot(x_2, m_2*x_2 + c_2)
    ax1[1].legend([f'Corr: {coeff_2:.3g}'])
    ax1[1].set_title('$\mathbf{E}[\\tau]\sim \exp(\lambda^{-1/2})$')
    ###
    # Learning rate
    (x, y, std, log_std, _) = lr_results
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax2[1].set_xlabel("$\eta$")
    ax2[1].set_ylabel("$\log(\mathbf{E}[\\tau])$")
    ax2[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax2[1].plot(x_2, m_2*x_2 + c_2)
    # ax2[1].set_ylim(bottom=0, top=None)
    ax2[1].legend([f'Corr: {coeff_2:.3g}'])
    ax2[1].set_title('$\mathbf{E}[\\tau]\sim \exp(\eta^{-1})$')
    #
    # Batch size
    (x, y, std, log_std, _) = batch_size_results
    #
    # coeff, _ = stats.pearsonr(x, y)
    # A = np.vstack([x, np.ones(len(x))]).T
    # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # ax3[0].set_xlabel("batch_size")
    # ax3[0].set_ylabel("exit time")
    # ax3[0].errorbar(x, y, yerr=std, fmt='.k') 
    # ax3[0].plot(x, m*x + c) 
    # # ax3[0].set_ylim(bottom=0, top=None)
    # ax3[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax3[1].set_xlabel("$B$")
    ax3[1].set_ylabel("$\log(\mathbf{E}[\\tau])$")
    ax3[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax3[1].plot(x_2, m_2*x_2 + c_2)
    # ax3[1].set_ylim(bottom=0, top=None)
    ax3[1].legend([f'Corr: {coeff_2:.3g}'])
    ax3[1].set_title('$\mathbf{E}[\\tau] \sim = \exp(B)$')
    # R
    (x, y, std, log_std, _) = r_results
    #
    # coeff, _ = stats.pearsonr(x, y)
    # A = np.vstack([x, np.ones(len(x))]).T
    # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # ax4[0].set_xlabel("r")
    # ax4[0].set_ylabel("exit time")
    # ax4[0].errorbar(x, y, yerr=std, fmt='.k') 
    # ax4[0].plot(x, m*x + c) 
    # # ax4[0].set_ylim(bottom=0, top=None)
    # ax4[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x**2
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax4[1].set_xlabel("$\Delta L$")
    ax4[1].set_ylabel("$\log(\mathbf{E}[\\tau])$")
    ax4[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax4[1].plot(x_2, m_2*x_2 + c_2)
    # ax4[1].set_ylim(bottom=0, top=None)
    ax4[1].legend([f'Corr: {coeff_2:.3g}'])
    ax4[1].set_title('$\mathbf{E}[\\tau] \sim \exp(\Delta L)$')
    # Log
    # x_3 = x
    # y_3 = np.sqrt(np.log(y))
    # coeff_3, _ = stats.pearsonr(x_3, y_3)
    # A = np.vstack([x_3, np.ones(len(x))]).T
    # m_3, c_3 = np.linalg.lstsq(A, y_3, rcond=None)[0]
    # ax4[2].set_xlabel("r")
    # ax4[2].set_ylabel("sqrt(log(exit time))")
    # ax4[2].errorbar(x_3, y_3, yerr=std*0, fmt='.k') 
    # ax4[2].plot(x_3, m_3*x_3 + c_3)
    # # ax4[2].set_ylim(bottom=0, top=None)
    # ax4[2].legend([f'Corr: {coeff_3:.3g}'])
    # ax4[2].set_title(f'tau = exp(r^2)')

    plt.tight_layout()
    plt.show()
    fig.savefig("results.png",dpi=300)

if __name__=='__main__':
    config_fn = 'MLP_SGD.json' 
    sharpness_results  = np.load(f"./{config_fn[:-5]}/sharpness_results.npy")
    lr_results         = np.load(f"./{config_fn[:-5]}/lr_results.npy")
    batch_size_results = np.load(f"./{config_fn[:-5]}/batch_size_results.npy")
    r_results          = np.load(f"./{config_fn[:-5]}/r_results.npy")
    draw(config_fn, sharpness_results, lr_results, batch_size_results, r_results)
