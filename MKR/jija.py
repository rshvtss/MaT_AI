package com.mycompany.code;

import java.util.ArrayList;
import java.util.List;

public class EditorUI {

    public static void main(String[] args) {
        List<Dot> dots = new ArrayList<>();

        IShapeFlyweight redCircle = ShapeFactory.getShapeType("Circle", "Red");
        dots.add(new Dot(10, 10, redCircle));
        dots.add(new Dot(20, 30, redCircle));
        dots.add(new Dot(40, 50, redCircle));

        IShapeFlyweight blueStar = ShapeFactory.getShapeType("Star", "Blue");
        dots.add(new Dot(100, 100, blueStar));
        dots.add(new Dot(120, 110, blueStar));

        IShapeFlyweight moreRedCircle = ShapeFactory.getShapeType("Circle", "Red");
        dots.add(new Dot(99, 99, moreRedCircle));

        System.out.println("\n--- Рендеринг сцени (6 об'єктів, але лише 2 типи у пам'яті) ---");
        for (Dot dot : dots) {
            dot.render();
        }
    }
}