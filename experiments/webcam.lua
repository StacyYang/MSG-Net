require 'torch'
require 'texture'
require 'image'
require 'camera'

require 'qt'
require 'qttorch'
require 'qtwidget'

local utils = require 'utils.utils'
local preprocess = require 'utils.preprocess'
local imgLoader = require 'utils.getImages'
local opts = require 'opts'

--[[ Model options
cmd:option('-models', 'models/instance_norm/candy.t7')
cmd:option('-height', 480)
cmd:option('-width', 640)

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)

-- Webcam options
cmd:option('-webcam_idx', 0)
cmd:option('-webcam_fps', 60)
--]]

local function main()
	local opt = opts.parse(arg)
	
	if (opt.input_image == '') then
    error('Must give exactly one of -input_image')
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

	local style_image = nil
	local function run_image(img, feat, idx)
		if opt.image_size > 0 then
			img = image.scale(img, opt.image_size)
		end
		local H, W = img:size(2), img:size(3)
		local img_pre = preprocess.preprocess(img:view(1, 3, H, W)):type(dtype)
	
		-- update style image
		if (idx-1) % 15 == 0 then
			local i=torch.floor((idx-1)/15)%styleLoader:size()+1
			style_image = styleLoader:get(i):float()
			model:setTarget(feat[i], dtype)
  	end

		local img_out = model:forward(img_pre)
		local styleSize = torch.floor(opt.image_size / 4) 
	
		style_image = image.scale(style_image,styleSize,styleSize)
		img_out = preprocess.deprocess(img_out:float())[1]
		img = img:float()
		img:sub(1,3,21,20+styleSize,21,20+styleSize):copy(style_image)
		img_out = torch.cat(img,img_out,3)

		if opt.median_filter > 0 then
	 		img_out = utils.median_filter(img_out, opt.median_filter)
 		end
		collectgarbage()
		collectgarbage()
 		return img_out
	end

  local camera_opt = {
    idx = opt.webcam_idx,
    fps = opt.webcam_fps,
    height = opt.height,
    width = opt.width,
  }
  local cam = image.Camera(camera_opt)

  local win = nil
	local idx = 1
	
  while true do
    -- Grab a frame from the webcam
    local img = cam:forward()

    -- Run the model
    local img_disp = run_image(img, feat, idx)

		idx = idx + 1
    if not win then
      -- On the first call use image.display to construct a window
      win = image.display(img_disp)
    else
      -- Reuse the same window
      win.image = img_disp
      local size = win.window.size:totable()
      local qt_img = qt.QImage.fromTensor(img_disp)
      win.painter:image(0, 0, size.width, size.height, qt_img)
    end
  end
end


main()

