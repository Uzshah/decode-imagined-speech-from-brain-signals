from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

class models:
    def knnclf(self, x_train, y_train, x_test, y_test):
        kclf = KNeighborsClassifier(n_neighbors=1)
        kclf.fit(x_train, y_train)
        y_pred = kclf.predict(x_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\n -------------Classification Report-------------\n")
        print(classification_report(y_test, y_pred))
        
    def rclf(self, x_train, y_train, x_test, y_test):
        rfc = RandomForestClassifier(max_depth=5)
        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\n -------------Classification Report-------------\n")
        print(classification_report(y_test, y_pred))
    
    
