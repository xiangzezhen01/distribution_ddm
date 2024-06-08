from DaL_Regressor import DaL_Regressor

train_index = [32, 173, 59, 167, 163, 191, 123, 22, 115, 169, 12, 7, 180, 58, 109, 13, 82, 29, 185, 177, 157, 106, 5, 95, 65, 100, 139, 39, 160, 162, 117, 51, 17, 165, 135, 94, 147, 137, 134, 155, 101, 99, 152, 71, 46]
test_index = [98, 107, 10, 66, 130, 124, 103, 77, 122, 91, 149, 55, 129, 35, 72, 178, 24, 158, 64, 136, 154, 37, 79, 25, 18, 84, 120, 143, 168, 90, 111, 80, 156, 52, 141, 183, 113, 133, 188, 15, 140, 3, 23, 102, 0, 126, 85, 62, 83, 16, 48, 56, 61, 36, 114, 181, 20, 81, 187, 125, 27, 184, 74, 31, 145, 104, 118, 69, 26, 148, 171, 70, 75, 138, 151, 11, 76, 159, 49, 40, 73, 30, 170, 172, 175, 105, 108, 4, 78, 166, 33, 60, 8, 116, 86, 96, 142, 19, 146, 189, 89, 186, 87, 50, 67, 176, 153, 110, 131, 119, 53, 179, 97, 57, 63, 45, 92, 41, 14, 144, 42, 190, 128, 2, 34, 161, 28, 47, 21, 121, 164, 174, 150, 6, 88, 9, 54, 44, 127, 68, 132, 112, 182, 38, 43, 93, 1]
dal = DaL_Regressor("data/Apache_AllNumeric.csv")
dal.get_cluster(dal.whole_data)
print("hello")
