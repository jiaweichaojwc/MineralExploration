using Pkg

# 激活当前目录的环境
Pkg.activate(".")

println("正在安装开发者工具 Revise...")
Pkg.add("Revise")

println("Revise 安装完成！")