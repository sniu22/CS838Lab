/**
 * @Author: Yuting Liu and Jude Shavlik.
 * <p>
 * Copyright 2017.  Free for educational and basic-research use.
 * <p>
 * The main class for Lab3 of cs638/838.
 * <p>
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 * <p>
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab3.java, insert that class here to simplify grading.
 * <p>
 * Some snippets from Jude's code left in here - feel free to use or discard.
 */

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;


public class Lab3 {

    private static int imageSize = 32;

    public static enum Category { airplanes, butterfly, flower, grand_piano, starfish, watch}

    protected static final double shiftProbNumerator = 6.0; // 6.0 is the 'default.'
    protected static final double probOfKeepingShiftedTrainsetImage = (shiftProbNumerator / 48.0);


    public static void loadDataset(Dataset dataset, File dir) {
        for (File file : dir.listFiles()) {
            // check all files
            if (!file.isFile() || !file.getName().endsWith(".jpg")) {
                continue;
            }
            //String path = file.getAbsolutePath();
            BufferedImage img = null, scaledBI = null;
            try {
                // load in all images
                img = ImageIO.read(file);
                // every image's name is in such format:
                // label_image_XXXX(4 digits) though this code could handle more than 4 digits.
                String name = file.getName();
                int locationOfUnderscoreImage = name.indexOf("_image");

                // Resize the image if requested.  Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
                if (imageSize != 128) {
                    scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = scaledBI.createGraphics();
                    g.drawImage(img, 0, 0, imageSize, imageSize, null);
                    g.dispose();
                }
                Instance instance = new Instance(scaledBI == null ? img : scaledBI, name, name.substring(0, locationOfUnderscoreImage));
                dataset.add(instance);
            } catch (IOException e) {
                System.err.println("Error: cannot load in the image file");
                System.exit(1);
            }
        }

    }



    public static int get_label(String label) { return Lab3.convertCategoryStringToEnum(label).ordinal(); }

    public static int get_prediction(double[] res) {
        int index = 0;
        double max_prediction = 0;
        for (int i = 0; i < res.length; i++) {
            if (res[i] > max_prediction) {
                index = i;
                max_prediction = res[i];
            }
        }
        return index;
    }

    public static Category convertCategoryStringToEnum(String name) {
        if ("airplanes".equals(name))
            return Category.airplanes; // Should have been the singular 'airplane' but we'll live with this minor error.
        if ("butterfly".equals(name)) return Category.butterfly;
        if ("flower".equals(name)) return Category.flower;
        if ("grand_piano".equals(name)) return Category.grand_piano;
        if ("starfish".equals(name)) return Category.starfish;
        if ("watch".equals(name)) return Category.watch;
        throw new Error("Unknown category: " + name);
    }


    private static final long millisecInMinute = 60000;
    private static final long millisecInHour = 60 * millisecInMinute;
    private static final long millisecInDay = 24 * millisecInHour;

    public static String convertMillisecondsToTimeSpan(long millisec) {
        int digits = 0;
        if (millisec == 0) {
            return "0 seconds";
        } // Handle these cases this way rather than saying "0 milliseconds."
        if (millisec < 1000) {
            return millisec + " milliseconds";
        } // Or just comment out these two lines?
        if (millisec > millisecInDay) {
            return millisec / millisecInDay + " days and " + convertMillisecondsToTimeSpan(millisec % millisecInDay);
        }
        if (millisec > millisecInHour) {
            return millisec / millisecInHour + " hours and " + convertMillisecondsToTimeSpan(millisec % millisecInHour);
        }
        if (millisec > millisecInMinute) {
            return millisec / millisecInMinute + " minutes and " + convertMillisecondsToTimeSpan(millisec % millisecInMinute);
        }

        return millisec / 1000.0 + " seconds";
    }



    private static void createMoreImages(Dataset trainset) {
        Dataset trainsetExtras = new Dataset();
        double probOfKeeping = 1.0;
        // Create New Images, Store them in trainsetExtra
        for (Instance trainImage : trainset.getImages()) {
            if (!"airplanes".equals(trainImage.getLabel()) &&  // Airplanes all 'face' right and up, so don't flip left-to-right or top-to-bottom.
                    !"grand_piano".equals(trainImage.getLabel())) {  // Ditto for pianos.

                if (trainImage.getProvenance() != Instance.HowCreated.FlippedLeftToRight && Tool.random() <= probOfKeeping)
                    trainsetExtras.add(trainImage.flipImageLeftToRight());

                // Butterflies all have the heads at the top, so don't flip to-to-bottom.
                // Ditto for flowers.
                // Star fish are standardized to 'point up.
                if (!"butterfly".equals(trainImage.getLabel()) && !"flower".equals(trainImage.getLabel()) && !"starfish".equals(trainImage.getLabel())) {
                    if (trainImage.getProvenance() != Instance.HowCreated.FlippedTopToBottom && Tool.random() <= probOfKeeping)
                        trainsetExtras.add(trainImage.flipImageTopToBottom());
                }
            }

            boolean rotateImages = true;
            if (rotateImages && trainImage.getProvenance() != Instance.HowCreated.Rotated) {
                //    Instance rotated = origTrainImage.rotateImageThisManyDegrees(3);
                //    origTrainImage.display2D(origTrainImage.getGrayImage());
                //    rotated.display2D(              rotated.getGrayImage()); waitForEnter();

                if (Tool.random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(3));
                if (Tool.random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(-3));
                if (Tool.random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(4));
                if (Tool.random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(-4));
                if (!"butterfly".equals(trainImage.getLabel()) &&  // Butterflies all have the heads at the top, so don't rotate too much.
                        !"flower".equals(trainImage.getLabel()) &&  // Ditto for flowers and starfish.
                        !"starfish".equals(trainImage.getLabel())) {
                    if (Tool.random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(5));
                    if (Tool.random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(-5));
                } else {
                    if (Tool.random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(2));
                    if (Tool.random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(-2));
                }
            }
            // Would be good to also shift and rotate the flipped examples, but more complex code needed.
            if (trainImage.getProvenance() != Instance.HowCreated.Shifted) {
                for (int shiftX = -3; shiftX <= 3; shiftX++) {
                    for (int shiftY = -3; shiftY <= 3; shiftY++) {
                        // Only keep some of these, so these don't overwhelm the flipped and rotated examples when down sampling below.
                        if ((shiftX != 0 || shiftY != 0) && Tool.random() <= probOfKeepingShiftedTrainsetImage * probOfKeeping)
                            trainsetExtras.add(trainImage.shiftImage(shiftX, shiftY));
                    }
                }
            }
        }


        // Add the extra image to trainset
        int count_trainsetExtrasKept = 0;
        int[] countOfCreatedTrainingImages = new int[Category.values().length];
        for (Instance createdTrainImage : trainsetExtras.getImages()) {
            // Trainset counts: airplanes=127, butterfly=55, flower=114, piano=61, starfish=51, watch=146
            if ("airplanes".equals(createdTrainImage.getLabel()))
                probOfKeeping = 0.4; // No flips, so fewer created.
            else if ("butterfly".equals(createdTrainImage.getLabel()))
                probOfKeeping = 0.4; // No top-bottom flips, so fewer created.
            else if ("flower".equals(createdTrainImage.getLabel()))
                probOfKeeping = 0.4; // No top-bottom flips, so fewer created.
            else if ("grand_piano".equals(createdTrainImage.getLabel()))
                probOfKeeping = 0.4; // No flips, so fewer created.
            else if ("starfish".equals(createdTrainImage.getLabel()))
                probOfKeeping = 0.4; // No top-bottom flips, so fewer created.
            else if ("watch".equals(createdTrainImage.getLabel()))
                probOfKeeping = 0.20; // Already have a lot of these.

            if (Tool.random() <= probOfKeeping) {
                countOfCreatedTrainingImages[convertCategoryStringToEnum(createdTrainImage.getLabel()).ordinal()]++;
                count_trainsetExtrasKept++;
                trainset.add(createdTrainImage);
            }
        }

//        for (Category cat : Category.values()) {
//            System.out.println(" Kept " + countOfCreatedTrainingImages[cat.ordinal()] + " images of " + cat + ".");
//        }
//        System.out.println("Created a total of " + trainsetExtras.getSize() + " new training examples and kept " + count_trainsetExtrasKept);
//        System.out.println("The trainset NOW contains " + trainset.getSize() + " examples.");

    }


    public static void main(String[] args) {
        String trainDirectory = "images/trainset/";
        String tuneDirectory = "images/tuneset/";
        String testDirectory = "images/testset/";

        // Here are statements with the absolute path to open images folder
        File trainsetDir = new File(trainDirectory);
        File tunesetDir = new File(tuneDirectory);
        File testsetDir = new File(testDirectory);

        // create three datasets
        Dataset trainset = new Dataset();
        Dataset tuneset = new Dataset();
        Dataset testset = new Dataset();

        // Load in images into datasets.
        long start = System.currentTimeMillis();


        loadDataset(trainset, trainsetDir);
        Lab3.createMoreImages(trainset);

        System.out.println("The trainset contains " + trainset.getSize() + " examples.");

        loadDataset(tuneset, tunesetDir);
        System.out.println("The  testset contains " + tuneset.getSize() + " examples.");

        loadDataset(testset, testsetDir);
        System.out.println("The  tuneset contains " + testset.getSize() + " examples.");

        System.out.println("\nLoad and generate data took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
        start = System.currentTimeMillis();
        Cnn.train_cph(trainset, tuneset, testset);
        System.out.println("\nTook " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to train.");


    }

}

