--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Created by: Hang Zhang
-- ECE Department, Rutgers University
-- Email: zhang.hang@rutgers.edu
-- Copyright (c) 2017
--
-- Free to reuse and distribute this software for research or 
-- non-profit purpose, subject to the following conditions:
--  1. The code must retain the above copyright notice, this list of
--     conditions.
--  2. Original authors' names are not deleted.
--  3. The authors' names are not used to endorse or promote products
--      derived from this software 
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

local M={}

function M.parse(arg)
	local cmd = torch.CmdLine()

	cmd:text()
	cmd:text('Options:')

	-- Generic options
	cmd:option('-model', 'hang')
	cmd:option('-model_nres', '2')
	cmd:option('-use_instance_norm', 1)
	cmd:option('-h5_file', 'data.h5')
	cmd:option('-padding_type', 'reflect-start')
	cmd:option('-tanh_constant', 150)
	cmd:option('-preprocessing', 'vgg')
	cmd:option('-resume', '')

	-- Style loss function options
	--cmd:option('-percep_loss_weight', 1.0)
	cmd:option('-tv_strength', 1e-6)

	-- Options for feature reconstruction loss
	cmd:option('-content_weights', '1.0')
	cmd:option('-content_layers', '16')
	cmd:option('-loss_network', 'models/vgg16.t7')

	-- Options for style reconstruction loss
	cmd:option('-style_image_folder', 'images/9styles')
	cmd:option('-style_image_size', 512)
	cmd:option('-style_iter', 20)
	cmd:option('-style_weights', '5.0')
	cmd:option('-style_layers', '4,9,16,23')

	-- Optimization
	cmd:option('-num_iterations', 800000)
	cmd:option('-max_train', -1)
	cmd:option('-batch_size', 4)
	cmd:option('-learning_rate', 1e-3)
	cmd:option('-lr_decay_every', -1)
	cmd:option('-lr_decay_factor', 0.5)
	cmd:option('-weight_decay', 0)

	-- Checkpointing
	cmd:option('-checkpoint_name', 'checkpoint')
	cmd:option('-checkpoint_every', 1000)
	cmd:option('-num_val_batches', 10)

	-- Backend options
	cmd:option('-gpu', 0)
	cmd:option('-use_cudnn', 1)
	cmd:option('-backend', 'cuda', 'cuda|opencl')

	-- Test options
	cmd:option('-premodel', 'models/model_9styles.t7')
	cmd:option('-input_image', 'images/content/venice-boat.jpg')
	cmd:option('-output_dir', 'stylized')
	cmd:option('-image_size', 512)
	cmd:option('-median_filter', 3)


	local opt = cmd:parse(arg or {})
	return opt
end

return M
