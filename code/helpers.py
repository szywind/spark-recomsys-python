import random

def generate_random_rating(clothes_file, user_file, rating_file):
    with open(clothes_file, 'r') as fl:
        clothes_id = [line.strip().split(',')[0] for line in fl]
    with open(user_file, 'r') as fl:
        user_id = [line.strip().split(',')[-1] for line in fl]
    N = 88
    f = open(rating_file, 'w')

    for i in range(N):
        item = random.sample(clothes_id, 1)
        user = random.sample(user_id, 1)
        rating = str(random.randrange(1,6))
        f.write(','.join([user[0][:user[0].rfind('.')], item[0][item[0].find('_')+1:item[0].rfind('.')], rating]) + '\n')
    f.close()

if __name__ == '__main__':
    generate_random_rating('../input/clothes-features.csv', '../input/user-features.csv', '../input/ratings')