require 'texture'
require 'image'

local utils = require 'utils.utils'
local preprocess = require 'utils.preprocess'
local imgLoader = require 'utils.getImages'
local opts = require 'opts'

local function main()
	local opt = opts.parse(arg)
	
  models = require('models/' .. opt.model)

	if (opt.input_image == '') and (opt.input_dir == '') then
    error('Must give exactly one of -input_image or -input_dir')
  end

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local ok, checkpoint = pcall(function() return torch.load(opt.premodel) end)
  if not ok then
    print('ERROR: Could not load model from ' .. opt.premodel)
    print('You may need to download the pretrained models by running')
    return
  end
  local model = checkpoint.model
  model:evaluate()
  model:type(dtype)  
  if use_cudnn then
    cudnn.convert(model, cudnn)
    if opt.cudnn_benchmark == 0 then
      cudnn.benchmark = false
      cudnn.fastest = true
    end
  end

  local preprocess_method = checkpoint.opt.preprocessing or 'vgg'
  local preprocess = preprocess[preprocess_method]
	local styleLoader = imgLoader(opt.style_image_folder)
	
	local featpath = opt.style_image_folder .. '/feat.t7'
	if not paths.filep(featpath) then
		local extractor = require "extractGram"
		extractor.exec(opt)
	end

	local feat = torch.load(featpath)
	feat = nn.utils.recursiveType(feat, 'torch.CudaTensor')

  local function run_image(in_path, out_dir, styleLoader, feat)
    if not path.isdir(out_dir) then
     	paths.mkdir(out_dir)
    end
    local img = image.load(in_path, 3)
    if opt.image_size > 0 then
      img = image.scale(img, opt.image_size)
    end
    local H, W = img:size(2), img:size(3)
    local img_pre = preprocess.preprocess(img:view(1, 3, H, W)):type(dtype)
    
		for i=1,styleLoader:size() do
			local style_image = styleLoader:get(i)
			model:setTarget(feat[i], dtype)
    
			if opt.timing == 1 then
      	-- Do an extra forward pass to warm up memory and cuDNN
      	model:forward(img_pre)
      	timer = torch.Timer()
      	if cutorch then cutorch.synchronize() end
    	end
			local img_out = model:forward(img_pre)
			if opt.timing == 1 then
     		if cutorch then cutorch.synchronize() end
      	local time = timer:time().real
      	print(string.format('Image %s (%d x %d) took %f',
            in_path, H, W, time))
    	end
			local img_out = preprocess.deprocess(img_out)[1]
    	if opt.median_filter > 0 then
      	img_out = utils.median_filter(img_out, opt.median_filter)
    	end

      local out_path = paths.concat(opt.output_dir, i) .. '.jpg'
			local out_path_style = paths.concat(opt.output_dir, i) .. 'style.jpg'
    	print('Writing output image to ' .. out_path)
			image.save(out_path, img_out)
    	image.save(out_path_style, style_image)
			collectgarbage()
			collectgarbage()
		end
  end

 if opt.input_image ~= '' then
    if opt.output_image == '' then
      error('Must give -output_image with -input_image')
    end
    run_image(opt.input_image, opt.output_dir, styleLoader, feat)
	else
		error('Must provide input image')
  end
end

main()
