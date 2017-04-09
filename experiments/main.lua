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

require 'texture'
require 'image'
require 'optim'

require 'utils.DataLoader'

local utils = require 'utils.utils'
local preprocess = require 'utils.preprocess'
local opts = require 'opts'
local imgLoader = require 'utils.getImages'

function main()
	local opt = opts.parse(arg)
  -- Parse layer strings and weights
  opt.content_layers, opt.content_weights =
    utils.parse_layers(opt.content_layers, opt.content_weights)
  opt.style_layers, opt.style_weights =
    utils.parse_layers(opt.style_layers, opt.style_weights)

  -- Figure out preprocessing
  if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
  end
  preprocess = preprocess[opt.preprocessing]

  -- Figure out the backend
  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

	-- Style images
	local styleLoader = imgLoader(opt.style_image_folder)
	local featpath = opt.style_image_folder .. '/feat.t7'
	if not paths.filep(featpath) then
		local extractor = require "extractGram"
		extractor.exec(opt)
	end
	local feat = torch.load(featpath)
	feat = nn.utils.recursiveType(feat, 'torch.CudaTensor')

  -- Build the model
  local model = nil

	-- Checkpoint
	if opt.resume ~= '' then
    print('Loading checkpoint from ' .. opt.resume)
    model = torch.load(opt.resume).model:type(dtype)
  else
  	print('Initializing model from scratch')
  	models = require('models/' .. opt.model)
		model = models.createModel(opt):type(dtype)
	end
  
	if use_cudnn then cudnn.convert(model, cudnn) end
  model:training()
  print(model)
  
  -- Set up the perceptual loss function
  local percep_crit
  local loss_net = torch.load(opt.loss_network)
  local crit_args = {
    cnn = loss_net,
    style_layers = opt.style_layers,
    style_weights = opt.style_weights,
    content_layers = opt.content_layers,
    content_weights = opt.content_weights,
  }
  percep_crit = nn.PerceptualCriterion(crit_args):type(dtype)

  local loader = DataLoader(opt)
  local params, grad_params = model:getParameters()

  local function f(x)
    assert(x == params)
    grad_params:zero()
    
    local x, y = loader:getBatch('train')
    x, y = x:type(dtype), y:type(dtype)

    -- Run model forward
    local out = model:forward(x)
		local target = {content_target=y}
    local loss = percep_crit:forward(out, target)
		local grad_out = percep_crit:backward(out, target)

    -- Run model backward
    model:backward(x, grad_out)

    return loss, grad_params
  end

  local optim_state = {learningRate=opt.learning_rate}
  local train_loss_history = {}
  local val_loss_history = {}
  local val_loss_history_ts = {}
  local style_loss_history = nil
    
	style_loss_history = {}
  for i, k in ipairs(opt.style_layers) do
    style_loss_history[string.format('style-%d', k)] = {}
  end
  for i, k in ipairs(opt.content_layers) do
    style_loss_history[string.format('content-%d', k)] = {}
  end

  local style_weight = opt.style_weight
  for t = 1, opt.num_iterations do
		-- set Target Here    
		if (t-1)%opt.style_iter == 0 then
			print('Setting Style Target')
			local idx = (t-1)/opt.style_iter % #feat + 1
			
			local style_image = styleLoader:get(idx)
  		style_image = image.scale(style_image, opt.style_image_size)
  		style_image = preprocess.preprocess(style_image:add_dummy())
    	percep_crit:setStyleTarget(style_image:type(dtype))

			local style_image_feat = feat[idx]
			model:setTarget(style_image_feat, dtype)
		end
    local epoch = t / loader.num_minibatches['train']

    local _, loss = optim.adam(f, params, optim_state)

    table.insert(train_loss_history, loss[1])

    for i, k in ipairs(opt.style_layers) do
      table.insert(style_loss_history[string.format('style-%d', k)],
      	percep_crit.style_losses[i])
    end
    for i, k in ipairs(opt.content_layers) do
      table.insert(style_loss_history[string.format('content-%d', k)],
        percep_crit.content_losses[i])
    end

    print(string.format('Epoch %f, Iteration %d / %d, loss = %f',
          epoch, t, opt.num_iterations, loss[1]), optim_state.learningRate)

    if t % opt.checkpoint_every == 0 then
      -- Check loss on the validation set
      loader:reset('val')
      model:evaluate()
      local val_loss = 0
      print 'Running on validation set ... '
      local val_batches = opt.num_val_batches
      for j = 1, val_batches do
        local x, y = loader:getBatch('val')
        x, y = x:type(dtype), y:type(dtype)
        local out = model:forward(x)
        --y = shave_y(x, y, out)
   
        local percep_loss = 0
        percep_loss = percep_crit:forward(out, {content_target=y})
        val_loss = val_loss + percep_loss
      end
      val_loss = val_loss / val_batches
      print(string.format('val loss = %f', val_loss))
      table.insert(val_loss_history, val_loss)
      table.insert(val_loss_history_ts, t)
      model:training()

      -- Save a checkpoint
      local checkpoint = {
        opt=opt,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_loss_history_ts=val_loss_history_ts,
        style_loss_history=style_loss_history,
      }
      local filename = string.format('%s.json', opt.checkpoint_name)
      paths.mkdir(paths.dirname(filename))
      utils.write_json(filename, checkpoint)

      -- Save a torch checkpoint; convert the model to float first
      model:clearState()
      if use_cudnn then
        cudnn.convert(model, nn)
      end
      model:float()
      checkpoint.model = model
      filename = string.format('%s.t7', opt.checkpoint_name)
      torch.save(filename, checkpoint)

      -- Convert the model back
      model:type(dtype)
      if use_cudnn then
        cudnn.convert(model, cudnn)
      end
      params, grad_params = model:getParameters()

			collectgarbage()
			collectgarbage()
    end

    if opt.lr_decay_every > 0 and t % opt.lr_decay_every == 0 then
      local new_lr = opt.lr_decay_factor * optim_state.learningRate
      optim_state = {learningRate = new_lr}
    end
  end
end


main()

