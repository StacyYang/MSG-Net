package = "texture"
version = "scm-1"

source = {
		url = "git://github.com/zhanghang1989/MSG-Net.git",
		tag = "master"
}

description = {
		summary = "Texture Master Network",
		detailed = [[
					Texture Master Network
					]],
		homepage = "https://github.com/zhanghang1989/MSG-Net"
}

dependencies = {
		"torch >= 7.0",
		"cutorch >= 1.0"
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      CMAKE_PREFIX_PATH="$(LUA_BINDIR)/..",
      CMAKE_INSTALL_PREFIX="$(PREFIX)"
   }
}
