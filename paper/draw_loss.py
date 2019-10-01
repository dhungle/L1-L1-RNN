import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 19})
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['figure.figsize'] = 9, 5

l1_path = 'l1.txt'
l2_path = 'l2.txt'
vanilla_path = 'vanilla.txt'
lstm_path = 'lstm.txt'
gru_path = 'gru.txt'

def find_psnr(path):
    ans = []
    magic_str = 'psnr'
    magic_len = len(magic_str)
    with open(path) as f:
        for lines in f:
            if magic_str in lines:
                start_idx = lines.index(magic_str) + magic_len + 1
                loss = lines[start_idx: start_idx + 7]
                ans.append(float(loss))
    return ans[:-1:25]

l1_loss = find_psnr(l1_path)
l2_loss = find_psnr(l2_path)
vanilla_loss = find_psnr(vanilla_path)
lstm_loss = find_psnr(lstm_path)
gru_loss = find_psnr(gru_path)
epoch = np.linspace(0, 201, 40)
fig = plt.figure()
plt.plot(epoch, l1_loss, '-', label='Our model', linewidth=4, color="black")
plt.plot(epoch, l2_loss, '-.', label='SISTA-RNN', linewidth=4, color="black")
plt.plot(epoch, vanilla_loss, '--', label='Stacked-RNN', linewidth=4, color="black")
plt.plot(epoch, lstm_loss, ':', label='Stacked-LSTM', linewidth=4, color="black")
plt.plot(epoch, gru_loss, linestyle=(0, (1, 1)), label='Stacked-GRU', linewidth=4, color="black")
plt.xlabel('Training epochs')
plt.xlim(0, 200)
plt.ylabel('PSNR (dB)')
plt.grid(ls='-.')
plt.legend()
plt.show()
fig.savefig('foo.pdf')




















