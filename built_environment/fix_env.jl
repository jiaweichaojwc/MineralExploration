using Pkg

# 激活当前目录的环境
Pkg.activate(".")

println("正在清理旧的编译文件和配置...")
# 如果存在可能导致冲突的 Manifest.toml，将其删除以强制基于新的 Project.toml 重新解析
if isfile("Manifest.toml")
    rm("Manifest.toml")
    println("已删除旧的 Manifest.toml")
end

println("正在解析并安装依赖包...")
# 重新安装包并更新注册表
Pkg.resolve()
Pkg.instantiate()

println("正在编译环境...")
# 预编译所有安装的包
Pkg.build()

println("环境修复完成！")