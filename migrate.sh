sudo chown sj851109:root ../aiis db.sqlite3
sudo rm -rf aicore/migrations/
python3 manage.py makemigrations aicore
python3 manage.py migrate aicore zero
python3 manage.py migrate aicore