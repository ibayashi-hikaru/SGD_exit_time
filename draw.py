
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
def draw(config_fn, sharpness_results, lr_results, batch_size_results, r_results):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 3, figsize=(12, 16))
    plt.suptitle(config_fn, x=0.05, y=1)
    # Sharpness 
    (x, y, std, log_std, _) = sharpness_results
    y += 0.00001
    coeff, _ = stats.pearsonr(x, y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    #
    ax1[0].set_xlabel("sharpness")
    ax1[0].set_ylabel("exit time")
    ax1[0].errorbar(x, y, yerr=std, fmt='.k') 
    ax1[0].plot(x, m*x + c) 
    # ax1[0].set_ylim(bottom=0, top=None)
    ax1[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax1[1].set_xlabel("sharpness")
    ax1[1].set_ylabel("log(exit time)")
    ax1[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax1[1].plot(x_2, m_2*x_2 + c_2)
    # ax1[1].set_ylim(bottom=0, top=None)
    ax1[1].legend([f'Corr: {coeff_2:.3g}'])
    ax1[1].set_title(f'tau = exp(sharpness)')
    #
    x_3 = x
    y_3 = np.log(y)**2
    coeff_3, _ = stats.pearsonr(x_3, y_3)
    A = np.vstack([x_3, np.ones(len(x_3))]).T
    m_3, c_3 = np.linalg.lstsq(A, y_3, rcond=None)[0]
    ax1[2].set_xlabel("sharpness")
    ax1[2].set_ylabel("log(exit time)^2")
    ax1[2].errorbar(x_3, y_3, yerr=std*0, fmt='.k') 
    ax1[2].plot(x_3, m_3*x_3 + c_3)
    # ax1[2].set_ylim(bottom=0, top=None)
    ax1[2].legend([f'Corr: {coeff_3:.3g}'])
    ax1[2].set_title(f'tau = exp(sharpness^(1/2))')
    ###
    # Learning rate
    (x, y, std, log_std, _) = lr_results
    y += 0.00001
    #
    coeff, _ = stats.pearsonr(x, y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    ax2[0].set_xlabel("lr")
    ax2[0].set_ylabel("exit time")
    ax2[0].errorbar(x, y, yerr=std, fmt='.k') 
    ax2[0].plot(x, m*x + c) 
    # ax2[0].set_ylim(bottom=0, top=None)
    ax2[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax2[1].set_xlabel("lr")
    ax2[1].set_ylabel("log(exit time)")
    ax2[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax2[1].plot(x_2, m_2*x_2 + c_2)
    # ax2[1].set_ylim(bottom=0, top=None)
    ax2[1].legend([f'Corr: {coeff_2:.3g}'])
    ax2[1].set_title(f'tau = exp(lr^(-1))')
    #
    x_3 = x
    y_3 = np.log(y)**2
    coeff_3, _ = stats.pearsonr(x_3, y_3)
    A = np.vstack([x_3, np.ones(len(x_3))]).T
    m_3, c_3 = np.linalg.lstsq(A, y_3, rcond=None)[0]
    ax2[2].set_xlabel("lr")
    ax2[2].set_ylabel("log(exit time)^2")
    ax2[2].errorbar(x_3, y_3, yerr=std, fmt='.k') 
    ax2[2].plot(x_3, m_3*x_3 + c_3)
    # ax2[2].set_ylim(bottom=0, top=None)
    ax2[2].legend([f'Corr: {coeff_3:.3g}'])
    ax2[2].set_title(f'tau = exp(lr^(-1/2))')
    # Batch size
    (x, y, std, log_std, _) = batch_size_results
    y += 0.00001
    #
    coeff, _ = stats.pearsonr(x, y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    ax3[0].set_xlabel("batch_size")
    ax3[0].set_ylabel("exit time")
    ax3[0].errorbar(x, y, yerr=std, fmt='.k') 
    ax3[0].plot(x, m*x + c) 
    # ax3[0].set_ylim(bottom=0, top=None)
    ax3[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax3[1].set_xlabel("batch size")
    ax3[1].set_ylabel("log(exit time)")
    ax3[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax3[1].plot(x_2, m_2*x_2 + c_2)
    # ax3[1].set_ylim(bottom=0, top=None)
    ax3[1].legend([f'Corr: {coeff_2:.3g}'])
    ax3[1].set_title(f'tau = exp(batch size)')
    # R
    (x, y, std, log_std, _) = r_results
    y += 0.00001
    #
    coeff, _ = stats.pearsonr(x, y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    ax4[0].set_xlabel("r")
    ax4[0].set_ylabel("exit time")
    ax4[0].errorbar(x, y, yerr=std, fmt='.k') 
    ax4[0].plot(x, m*x + c) 
    # ax4[0].set_ylim(bottom=0, top=None)
    ax4[0].legend([f'Corr: {coeff:.3g}'])
    # Log
    x_2 = x**2
    y_2 = np.log(y)
    coeff_2, _ = stats.pearsonr(x_2, y_2)
    A = np.vstack([x_2, np.ones(len(x))]).T
    m_2, c_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
    ax4[1].set_xlabel("$\Delta L$")
    ax4[1].set_ylabel("log(exit time)")
    ax4[1].errorbar(x_2, y_2, yerr=log_std, fmt='.k') 
    ax4[1].plot(x_2, m_2*x_2 + c_2)
    # ax4[1].set_ylim(bottom=0, top=None)
    ax4[1].legend([f'Corr: {coeff_2:.3g}'])
    ax4[1].set_title(f'$\\tau = \exp(\Delta L)$')
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
