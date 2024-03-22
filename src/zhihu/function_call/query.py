# 参考：
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
from zhihu.common.api import Session
from zhihu.function_call.common import model_func_call, make_func_tool, function_calling_cb, DB, DBAnalyzer
import logging

logger = logging.getLogger(__name__)

class Order(DB, DBAnalyzer):
    tbl_exists = "SELECT name FROM sqlite_master WHERE type='table' AND name='{tbl_name}'"
    create_tbl_orders = """
    CREATE TABLE orders (
        id INT PRIMARY KEY NOT NULL,   -- 主键, 不允许为空
        customer_id INT NOT NULL,      -- 客户ID, 不允许为空
        product_id STR NOT NULL,       -- 产品ID, 不允许为空
        price DECIMAL(10, 2) NOT NULL, -- 价格, 不允许为空
        status INT NOT NULL,           -- 订单状态, 不允许为空，0代表待支付,1代表已支付, 2代表已退款
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间,默认为s前时间
        pay_time TIMESTAMP            -- 支付时间, 可以为空
    );
    """
    mock_data = [
        (1, 1001, "TSHIRT_1", 50.00, 0, '2023-10-12 10:00:00', None),
        (2, 1001, 'TSHIRT_2', 75.50, 1, '2023-10-16 11:00:00', '2023-08-16 12:00:00'),
        (3, 1002, 'SHOES_X2', 25.25, 2, '2023-10-17 12:30:00', '2023-08-17 13:00:00'),
        (4, 1003, 'HAT_Z112', 60.75, 1, '2023-10-20 14:00:00', '2023-08-20 15:00:00'),
        (5, 1002, 'WATCH_X001', 90.00, 0, '2023-10-28 16:00:00', None)
    ]
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Order, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.database_schema_string = self.create_tbl_orders
        super().__init__(self, database_schema_string=self.create_tbl_orders)
        # create the order table
        self.cursor.execute(self.tbl_exists.format(tbl_name="orders"))
        table_exists = self.cursor.fetchone() is not None
        if not table_exists:
            self._create_order_tbl()
            self._insert_order_tbl()
        self.conn.commit()
    
    def _drop_order_tbl(self):
        self.cursor.execute("DROP TABLE orders")
    
    def _create_order_tbl(self):
        self.cursor.execute(self.create_tbl_orders)

    def _insert_order_tbl(self):
        for record in self.mock_data:
            self.cursor.execute("INSERT INTO orders (id, customer_id, product_id, price, status, create_time, pay_time) VALUES (?,?,?,?,?,?,?)", record)

    def __del__(self):
        self._drop_order_tbl()
        super().__del__()


def exec_query(**args):
    logger.info("query=\n\n\033[34m%s\033[0m\n\n", args["query"])
    order_tbl = Order()
    return order_tbl.exec_query(args["query"])

def analyze(prompt, system_prompt):
    order = Order()
    return order.analyze(prompt, system_prompt, caller_module_name=__name__)