import os
import warnings
import torch
import torch.nn as nn
import utils as utility
import dataloaders.sad as sad
import nets.iwsr_dnn_hmm as iwsr

def _main(args):

	warnings.filterwarnings("ignore") 

	##### Analysis Number #####
	analysis_num = args.analysis

	##### Constructing Data Loader #####
	eval_data_root, preprcessed_data_root, eval_num_derivative, num_workers=  (
															args.dataset, 
															'{}/analysis_{}'.format(args.preprocessing_path, analysis_num), 
															args.num_derivative,
															args.num_workers
														)
	val_data_loader = sad.loader( data_root='{}/val'.format(eval_data_root), num_derivative=eval_num_derivative)
	test_data_loader = sad.loader( data_root='{}/test'.format(eval_data_root), num_derivative=eval_num_derivative)

	##### Loading Model #####
	report_path = './reports/{}'.format(analysis_num)
	model_path = '{}/models'.format(report_path)
	start_epoch, num_epochs = args.start_epoch, args.num_epochs
	last_epoch = num_epochs + start_epoch - 1

	##### Defining Criterion #####
	criterion = None
	if args.criterion=='mse':
		criterion = nn.MSELoss()
	elif args.criterion=='cross_entropy':
		criterion = nn.CrossEntropyLoss()

	print('{} Validation {}'.format('-'*32, '-'*32) )

	for epoch in range(start_epoch, start_epoch+num_epochs):
		print('{} epoch={} {}'.format('='*32, epoch, '='*32) )
		model = iwsr.load(model_path, 'iwsr_dnn_hmm_epoch_{}'.format(epoch))
		if args.gpu and torch.cuda.is_available():
			model=model.cuda()
		iwsr.eval(  model,
					preprcessed_data_root,
					val_data_loader,
					criterion,
					report_path,
					eval_mode='val',
					num_workers=num_workers,
					gpu= args.gpu and torch.cuda.is_available(),
					preprocess=args.preprocess,
					epoch=epoch,
					eval_num_derivative=eval_num_derivative
				)

	print('{} Test {}'.format('-'*32, '-'*32) )

	model = iwsr.load(model_path, 'iwsr_dnn_hmm_epoch_{}'.format(last_epoch))
	if args.gpu and torch.cuda.is_available():
		model=model.cuda()

	##### Evaluating #####
	iwsr.eval(  model,
				preprcessed_data_root,
				test_data_loader,
				criterion,
				report_path,
				eval_mode='test',
				num_workers=num_workers,
				gpu= args.gpu and torch.cuda.is_available(),
				preprocess=True,
				eval_num_derivative=eval_num_derivative
			 )

if __name__ == "__main__":
	args = utility.get_args()
	_main(args)