package flinkClassifiersTesting.inputs;

public class PlainExample extends BaseExample {
    private String plainClass;

    public PlainExample() {
    }


    public String getPlainClass() {
        return plainClass;
    }

    public void setPlainClass(String plainClass) {
        this.plainClass = plainClass;
    }

    @Override
    public String toString() {
        return "PlainExample{" +
                "plainClass='" + plainClass + '\'' +
                '}';
    }
}
