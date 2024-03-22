from zhihu.function_call.common import DB
from zhihu.function_call.query import Order

class MultiTable(DB):
    tbl_exists = "SELECT name FROM sqlite_master WHERE type='table' AND name='{tbl_name}'"
    create_tbl_customers = """
    CREATE TABLE customers (
        id INT PRIMARY KEY NOT NULL,  -- 主键，不允许为空
        customer_name VARCHAR(255) NOT NULL,  -- 客户姓名，不允许为空
        email VARCHAR(255) UNIQUE,  -- 邮箱，唯一
        register_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 注册时间，默认当前时间
    );
    """
    create_tbl_products = """
    CREATE TABLE products (
        id INT PRIMARY KEY NOT NULL,  -- 主键，不允许为空
        product_name VARCHAR(255) NOT NULL,  -- 产品名称，不允许为空
        price DECIMAL(10, 2) NOT NULL  -- 价格，不允许为空
    );
    """
    create_tbl_orders = Order.create_tbl_orders
    mock_data_customers = [
        (1001, "王卓然", "wangzhuoran@163.colm", "2023-10-12 10:00:00"),
        (1002, "孙志刚", "sunzhigang@126.com", "2023-10-16 11:00:00"),
        (1003, "李晓明", "lixiaoming@hotmail.com", "2023-10-17 12:30:00")
    ]
    mock_data_products = [
        (2001, "TSHIRT_1", 50.00),
        (2002, "TSHIRT_2", 75.50),
        (2003, "SHOES_X2", 25.25),
        (2004, "HAT_Z112", 60.75),
        (2005, "WATCH_X001", 90.00),
    ]
    # this mock_data_orders is different from that of Order's
    mock_data_orders = [
        (1, 1001, 2001, 50.00, 0, "2023-10-12 10:00:00", None),
        (2, 1001, 2002, 75.50, 1, "2023-10-16 11:00:00", "2023-08-16 12:00:00"),
        (3, 1002, 2003, 25.25, 2, "2023-10-17 12:30:00", "2023-08-17 13:00:00"),
        (4, 1003, 2004, 60.75, 1, "2023-10-20 14:00:00", "2023-08-20 15:00:00"),
        (5, 1002, 2005, 90.00, 0, "2023-10-28 16:00:00", None),
        (6, 1002, 2002, 75.50, 1, "2023-10-16 12:00:00", "2023-10-19 12:00:00"),
    ]
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MultiTable, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self):
        super().__init__()
        for name in ["customers", "products", "orders"]:
           if not self._tbl_exist(name):
                self._create_tbl(name)
                self._insert_data(name)
        self.conn.commit()
    
    def __del__(self):
        self._drop_tables(["customers", "products", "orders"])
        super().__del__()

    def _tbl_exist(self, name):
        self.cursor.execute(self.tbl_exists.format(tbl_name=name))
        return self.cursor.fetchone() is not None
    
    def _create_tbl(self, name):
        create_tbl_schema = getattr(self, f"create_tbl_{name}")
        self.cursor.execute(create_tbl_schema)
    
    def _insert_data(self, name):
        mock_data = getattr(self, f"mock_data_{name}")
        for record in mock_data:
            self.cursor.execute(f"INSERT INTO {name} VALUES ({','.join(['?']*len(record))})", record)
    
    def _drop_tbl(self, names):
        for name in names:
            if self._tbl_exist(name):
                self.cursor.execute(f"DROP TABLE {name}")