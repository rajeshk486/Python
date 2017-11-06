import pymysql

class DbConnection:

    def __int__(self):
        global TABLENAME

        global conn
        global cur

        TABLENAME= "rajesh"
        conn = pymysql.connect(host='localhost', port=3306, user='dbuser', passwd='Dbuser@123', db='mysql',  autocommit=True)
        cur = conn.cursor()
        cur.execute("use ML")
        stmt = "SHOW TABLES LIKE '" + TABLENAME + "'"

    def Table_exist(self):
        TABLENAME = "candidate"
        conn = pymysql.connect(host='localhost', port=3306, user='dbuser', passwd='Dbuser@123', db='mysql', autocommit=True)
        cur = conn.cursor()
        stmt = "SHOW TABLES LIKE '"+TABLENAME+"'"
        cur.execute(stmt)
        result = cur.fetchone()
        if not result:
            stmt = "CREATE TABLE"+TABLENAME+"(Path VARCHAR(250), Skills VARCHAR(2500))"

    def Insert_Table(self,values1,values2):
        conn = pymysql.connect(host='localhost', port=3306, user='dbuser', passwd='Dbuser@123', db='ML',
                               cursorclass=pymysql.cursors.DictCursor,autocommit=True)
        cur = conn.cursor()
        # cur.execute("use ML;")
        stmt = "INSERT INTO candidate values('"+values1+"','"+values2+"');"
        number_of_rows = cur.execute(stmt)
        conn.commit()
        result = cur.fetchone()
        if result:
            return True
        else:
            return False