/**
 * Author: Shuo Niu
 * Email: sniu22@wisc.edu
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Lab1 is used to input data from the given format.
 * It assumes two classes for each variables.
 */

public class Lab1 {

    public int varSize;
    public int sampleSize;
    public List var = new ArrayList();
    public List classValue = new ArrayList();
    public String[][] varValue;
    public String[][] sample;
    public List sampleCalss = new ArrayList();
    public int[] classCount;
    private boolean varIn = false;

    public Lab1(String filepath) {

        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(filepath));
        } catch (FileNotFoundException e) {
            System.err.println("Could not find file '" + filepath +
                    "'.");
            System.exit(1);
        }
        // Iterate through each line in the file.
        int lineCount = 0, varRec = 0, others = 0, sampleRec = 0;
        while (fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();
            // Skip blank lines.
            if (line.length() == 0 | line.startsWith("//")) {
                continue;
            } else lineCount++;

            if (lineCount == 1) {
                this.varSize = Integer.parseInt(line);
                this.varValue = new String[varSize][];
                this.varIn = true;
                continue;
            }

            if (this.varIn) {
                String[] lineVar = line.split(" - ");
                this.var.add(varRec, lineVar[0]);
                String[] varV = lineVar[1].split(" ");
                this.varValue[varRec] = new String[varV.length];
                this.varValue[varRec] = varV;
                varRec++;
                if (varRec >= this.varSize) {
                    this.varIn = false;
                }
                continue;
            }

            if (others < 3) {
                if (others <= 1) {
                    this.classValue.add(line);
                }
                if (others == 2) {
                    this.sampleSize = Integer.parseInt(line);
                    this.sample = new String[this.sampleSize][];
                }
                others++;
                continue;
            }
            if (sampleRec < sampleSize) {
                String pattern = "^(\\w*) (\\w*) (.*)$";
                Pattern r = Pattern.compile(pattern);
                Matcher m = r.matcher(line);
                if (m.find()) {
                    this.sampleCalss.add(m.group(2));
                    String[] thisVar = m.group(3).split(" [ ]*");
                    this.sample[sampleRec] = new String[this.varSize];
                    this.sample[sampleRec] = thisVar;
                    sampleRec++;
                }

            }

        }
        this.classCount = new int[this.classValue.size()];
        for (int i = 0; i < sampleSize; i++) {
            classCount[classValue.lastIndexOf(sampleCalss.get(i))] ++;
        }
    }

    public int[] count(int varPos) {
        int[] vcount = new int[2];
        for (int i = 0; i < this.sampleSize; i++) {
            boolean label = this.sampleCalss.get(i).equals(this.classValue.get(0));
            boolean ifFirst = this.sample[i][varPos].equals(this.varValue[varPos][0]);
            if(label && ifFirst){
                vcount[0]++;
            }
            else if (!label && ifFirst){
                vcount[1]++;
            }
        }
        return vcount;
    }

    public String summary() {
        String words = "There are " + this.varSize + " features in the dataset.\n";
        words += "There are " + this.sampleSize + " examples.\n";
        words += this.classCount[0] + " have output label '" + this.classValue.get(0) + "', " +
                this.classCount[1] + " have output label '"+  this.classValue.get(1) + "'.\n\n";

        for (int i = 0; i < this.var.size(); i++) {
            int[] scount = this.count(i);
            words += "Feature '" + this.var.get(i) + "':\n";
            String part0 = String.format("%.1f", (double) scount[0] / this.classCount[0] * 100);
            String part1 = String.format("%.1f", (double) scount[1] / this.classCount[0] * 100);
            words +=  "  In the examples with output label '" + this.classValue.get(0) + "', " +
                    part0 + "% have value '" + this.varValue[i][0] + "'\n";
            words +=  "  In the examples with output label '" + this.classValue.get(1) + "', " +
                    part1 + "% have value '" + this.varValue[i][0] + "'\n\n";
        }
        return words;
    }

    public static void main(String[] args) {
        Lab1 train = new Lab1(args[0]);
        Lab1 tune = new Lab1(args[1]);
        Lab1 test = new Lab1(args[2]);
        Perceptron testmodel = new Perceptron(train, tune, test, 0.1);

        testmodel.trainModel();

        System.out.println(testmodel.predict());


    }

}


/**
 * Perceptron is a naive implementation of perceptron method.
 * It uses the sigmoid activation function.
 */

class Perceptron {
    // Use the squared loss function as loss function, sigmoid as activation function
    public int inputNum;
    public double[] weight;
    public double step;
    public int[][] train, test, tune;
    public int[] trainC, testC, tuneC;
    public List label = new ArrayList();

    public void toNumeric(Lab1 strain, Lab1 stune, Lab1 stest) {
        //train set
        this.train = new int[strain.sampleSize][this.inputNum];
        for (int i = 0; i < strain.sampleSize; i++) {
            for (int j = 0; j < strain.varSize; j++) {
                this.train[i][2 * j] = (strain.sample[i][j].equals(strain.varValue[j][0])) ? 1 : 0;
                this.train[i][2 * j + 1] = (strain.sample[i][j].equals(strain.varValue[j][1])) ? 1 : 0;
            }
        }
        this.trainC = new int[strain.sampleSize];
        for (int i = 0; i < strain.sampleSize; i++) {
            this.trainC[i] = (strain.sampleCalss.get(i).equals(strain.classValue.get(0))) ? 1 : 0;
        }

        this.tune = new int[stune.sampleSize][this.inputNum];
        for (int i = 0; i < stune.sampleSize; i++) {
            for (int j = 0; j < strain.varSize; j++) {
                this.tune[i][2 * j] = (stune.sample[i][j].equals(strain.varValue[j][0])) ? 1 : 0;
                this.tune[i][2 * j + 1] = (stune.sample[i][j].equals(strain.varValue[j][1])) ? 1 : 0;
            }
        }
        this.tuneC = new int[stune.sampleSize];
        for (int i = 0; i < stune.sampleSize; i++) {
            this.tuneC[i] = (stune.sampleCalss.get(i).equals(strain.classValue.get(0))) ? 1 : 0;
        }

        this.test = new int[stest.sampleSize][this.inputNum];
        for (int i = 0; i < stest.sampleSize; i++) {
            for (int j = 0; j < strain.varSize; j++) {
                this.test[i][2 * j] = (stest.sample[i][j].equals(strain.varValue[j][0])) ? 1 : 0;
                this.test[i][2 * j + 1] = (stest.sample[i][j].equals(strain.varValue[j][1])) ? 1 : 0;
            }
        }
        this.testC = new int[stest.sampleSize];
        for (int i = 0; i < stest.sampleSize; i++) {
            this.testC[i] = (stest.sampleCalss.get(i).equals(strain.classValue.get(0))) ? 1 : 0;
        }

    }

    public Perceptron(Lab1 train, Lab1 tune, Lab1 test, double step) {
        inputNum = 0;
        for (int i = 0; i < train.varValue.length; i++) {
            inputNum += train.varValue[i].length;
        }
        this.weight = new double[inputNum + 1];
        for (int i = 0; i <= inputNum; i++) {
            weight[i] = (Math.random() - 0.5) / 100;
        }
        this.step = step;
        this.toNumeric(train, tune, test);
        this.label = train.classValue;
    }

    public double output(int[] sample) {
        double out = 0.0;
        for (int i = 0; i < weight.length - 1; i++) {
            out += sample[i] * weight[i];
        }
        out += weight[weight.length - 1];
        return 1 / (1 + Math.exp(-1 * out));
    }

    public void update(int[] sample, int sampleClass) {
        double out = this.output(sample);
        for (int i = 0; i < sample.length - 1; i++) {
            this.weight[i] += this.step * out * (1 - out) * (sampleClass - out) * sample[i];
        }
        this.weight[sample.length - 1] += this.step * out * (1 - out) * (sampleClass - out);
    }

    public double[] accuracy() {
        double[] correct = new double[3];
        for (int i = 0; i < this.train.length; i++) {
            if (Math.abs(this.output(this.train[i]) - this.trainC[i]) < 0.5) correct[0] += 1;
        }
        for (int i = 0; i < this.tune.length; i++) {
            if (Math.abs(this.output(this.tune[i]) - this.tuneC[i]) < 0.5) correct[1] += 1;
        }
        for (int i = 0; i < this.test.length; i++) {
            if (Math.abs(this.output(this.test[i]) - this.testC[i]) < 0.5) correct[2] += 1;
        }
        correct[0] /= this.train.length;
        correct[1] /= this.tune.length;
        correct[2] /= this.test.length;

        return correct;
    }

    public void trainModel() {
        double last = -1, now = 0;
        while (now > last) {
            for (int i = 0; i < this.train.length; i++) {
                this.update(this.train[i], this.trainC[i]);
            }
            last = now;
            double[] result = this.accuracy();
            now = result[1];
        }
    }

    public String predict() {
        String res = "Prediction for test data. \n";
        double correct = 0;
        for (int i = 0; i < this.test.length; i++) {
            int real = this.testC[i];
            int pre = 0;
            if (this.output(this.test[i]) >= 0.5) pre = 1;
            int num = i + 1;
            res += "Sample" + num + " : the true class is " + this.label.get(real) +
                    " , the predicted class is " + this.label.get(pre) + "\n";
            if (real == pre) correct += 1;
        }
        correct = correct / this.test.length * 100;
        res += "The overall accuracy is :" + correct + "%.";

        return res;
    }

}



