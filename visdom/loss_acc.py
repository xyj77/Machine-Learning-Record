import numpy as np
import visdom
import os

default_path = './'

def plot_loss_acc(record_path = default_path):
	files = os.listdir(record_path)
	viz = visdom.Visdom()

	for file in files:
		file_name_list = file.split('_')
		if(os.path.splitext(file)[1] == '.txt' and len(file_name_list) > 2 and file_name_list[0] == 'train'):
			train_file = os.path.join(record_path, 'train_' + '_'.join(file_name_list[1:]))
			test_file = os.path.join(record_path, 'test_' + '_'.join(file_name_list[1:]))
			train_record = np.loadtxt(train_file)
			test_record = np.loadtxt(test_file)
			epoch_len = len(train_record)

			if('loss' in file_name_list[2] and file_name_list[1] == 'AE'): 
				viz.env = 'loss_AE' 
			elif('loss' in file_name_list[2] and file_name_list[1] == 'FullConnect'):
				viz.env = 'loss_FC'
			elif('acc' in file_name_list[2]):
				viz.env = 'acc'

			line = viz.line(X=np.column_stack((np.array(range(epoch_len)), np.array(range(epoch_len)))),
				            Y=np.column_stack((np.array(train_record), np.array(test_record))),
				            opts= dict(
				                legend=["train_loss", "test_loss"], 
				                xlabel='epoch', 
				                title='_'.join(file_name_list[1:3])
				                ))
			print(file + 'has been ploted.') 

if __name__ == '__main__':
	try:
		record_path = sys.argv[1]
		plot_loss_acc(record_path)
	except:
		plot_loss_acc()
