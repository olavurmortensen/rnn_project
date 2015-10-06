import MySQLdb

def execute_updates(host, database, user, password, sql_list):
    if len(sql_list)== 0:
        return
    config = {
            'host': host,
            # 'host': '62.61.146.181',
            'port': 3306,
            'db': database,
            'user': user,
            'passwd': password,
            'charset': 'utf8'
        }
    db = MySQLdb.Connect(**config)
    cursor = db.cursor()
    db.autocommit(False)
    for sql in sql_list:
        cursor.execute(sql)
    db.commit()

    cursor.close()
    db.close()


class SimpleDatabaseIterator:
    db_connections = {}

    def __init__(self, host, database, user, password, sql):
        self.sql = sql

        self.config = {
            'host': host,
            # 'host': '62.61.146.181',
            'port': 3306,
            'db': database,
            'user': user,
            'passwd': password,
            'charset': 'utf8'
        }

        if (host, database) not in SimpleDatabaseIterator.db_connections:
            SimpleDatabaseIterator.db_connections[(host, database)] = MySQLdb.Connect(**self.config)



        def generate_rows(sql):

            # db = MySQLdb.Connect(**self.config)
            # cursor = db.cursor()

            # if SimpleDatabaseIterator.db == None:
            #     SimpleDatabaseIterator.db = MySQLdb.Connect(**self.config)
            current_connection = SimpleDatabaseIterator.db_connections[(self.config['host'], self.config['db'])]
            cursor = current_connection.cursor()

            cursor.execute(sql)
            num_fields = len(cursor.description)
            field_names = [i[0] for i in cursor.description]
            counter = 0
            for row in cursor:
                row_vals = {}
                for i, field in enumerate(row):
                    row_vals[field_names[i]] = field
                yield row_vals
                counter+=1
                # if counter%10000==0:
                #     print row_vals
            cursor.close()
            # db.close()
        self.generator = generate_rows(sql)

    def __iter__(self):
        return self



    def next(self):
        return self.generator.next()

# db_iter = SimpleDatabaseIterator('findzebra', 'root', 'findzebra', 'select * from contexts2')
# for i in range(20):
#     print db_iter.next()

