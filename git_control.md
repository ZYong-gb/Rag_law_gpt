git init

eval "$(ssh-agent -s)"
ssh-add ~/user_key/zy_github_key
password:zy1314520..

git config user.email "1603664660@qq.com"
git config user.name "ZYong-gb"


git add .

git commit -m "first commit"

git branch -M main

git remote add origin git@github.com:ZYong-gb/Rag_law_gpt.git

git push -u origin main

git pull origin main