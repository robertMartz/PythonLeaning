class RegisterBox:
    def __init__(self, filename):
        self.data = {
            "date": [],
            "top_product": [],
            "income": [],
            "expenses": [],
            "payment_method": [],
            "revenue": []
        }

        self.number_of_days = 0
        self.total_income = 0
        self.total_revenue = 0

        self.max_income = 0
        self.min_income = 1e+10
        self.max_income_date = ""
        self.min_income_date = ""

        self.avg_revenue = 0
        self.max_revenue = 0
        self.min_revenue = 1e+10
        self.max_revenue_date = ""
        self.min_revenue_date = ""

        self.card_total = 0
        self.cash_total = 0

        with open(filename) as file:
            lines = file.readlines()
            index = 0
            for line in lines[1:]:
                line_info = line.strip().split(",")
                self.data["date"].append(line_info[0])
                self.data["top_product"].append(line_info[1])
                self.data["income"].append(float(line_info[2]))
                self.data["expenses"].append(float(line_info[3]))
                self.data["payment_method"].append(line_info[4])
                self.data["revenue"].append(float(line_info[2]) - float(line_info[3]))

                self.total_income += self.data["income"][index]
                self.total_revenue += self.data["revenue"][index]

                if self.data["income"][index] > self.max_income:
                    self.max_income = self.data["income"][index]
                    self.max_income_date = self.data["date"][index]
                if self.data["income"][index] < self.min_income:
                    self.min_income = self.data["income"][index]
                    self.min_income_date = self.data["date"][index]

                if self.data["revenue"][index] > self.max_revenue:
                    self.max_revenue = self.data["revenue"][index]
                    self.max_revenue_date = self.data["date"][index]
                if self.data["revenue"][index] < self.min_revenue:
                    self.min_revenue = self.data["revenue"][index]
                    self.min_revenue_date = self.data["date"][index]

                if self.data["payment_method"][index] == "Tarjeta":
                    self.card_total += self.data["income"][index]
                else:
                    self.cash_total += self.data["income"][index]

                index += 1

            self.number_of_days = index + 1

        self.avg_revenue = self.total_revenue / self.number_of_days
        day_type = []
        for i in self.data["revenue"]:
            if i > (self.avg_revenue * 1.5):
                day_type.append("Great")
            elif i < (self.avg_revenue * 0.8):
                day_type.append("Bad")
            else:
                day_type.append("Average")

        self.data["day_type"] = day_type

    def report_day_types(self):
        day_type_count = {
            "Great": 0,
            "Bad": 0,
            "Average": 0,
        }

        for i in self.data["day_type"]:
            if i == "Great":
                day_type_count["Great"] += 1
            elif i == "Bad":
                day_type_count["Bad"] += 1
            else:
                day_type_count["Average"] += 1

        print(f"There were {day_type_count['Great']} great days, {day_type_count['Bad']} bad days and "
              f"{day_type_count['Average']} average days in this period")

    def report_top_product(self):
        product_count = {}
        for i in self.data["top_product"]:
            if i not in product_count:
                product_count[i] = 1
            else:
                product_count[i] += 1
        #################################################### CONTINUAAAAAAAAAAAAAA AQUIIIIIIIIIIIIIIIIIII

    def make_report(self):
        print(f"Days in the period: {self.number_of_days}")
        print(f"Total income: {self.total_income:.2f} ")
        print(f"Total revenue: {self.total_revenue:.2f} ")
        print(f"Average revenue: {self.avg_revenue:.2f}")

        print(f"Total income by card: {self.card_total:.2f} ")
        print(f"Total income by cash: {self.cash_total:.2f} ")

        print(f"Most profitable day {self.max_revenue_date} with income: {self.max_income:.2f}")
        print(f"Least profitable day {self.min_revenue_date} with income: {self.min_income:.2f}")

        self.report_day_types()

filepath = input("Please enter the file path: ")
register_box = RegisterBox(filepath)
register_box.make_report()











