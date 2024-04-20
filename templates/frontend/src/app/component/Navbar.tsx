"use client";
import { buttonVariants, Button } from "../../components/ui/button";
import Link from "next/link";
import cn from "../lib/utils";
import * as React from "react";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { Github } from "lucide-react";
import Image from "next/image";
import logo from "../../assets/logo.png";

const components: { title: string; href: string; description: string }[] = [
  {
    title: "Data Transformation",
    href: "/preprocessing#transformation",
    description:
      "A modal dialog that interrupts the user with important content and expects a response.",
  },
  {
    title: "Explaination Histogram",
    href: "/preprocessing#ex_hist",
    description:
      "For sighted users to preview content available behind a link.",
  },
  {
    title: "Data Augmentation",
    href: "/preprocessing#data_aug",
    description:
      "Displays an indicator showing the completion progress of a task, typically displayed as a progress bar.",
  },
  {
    title: "Image Enhancement",
    href: "/preprocessing#img_en",
    description: "Visually or semantically separates content.",
  },
];

const Navbar = () => {
  return (
    <nav className="lex-no-wrap fixed top-0 h-[55px] w-full z-30 shadow-md shadow-black/5 bg-white transition-all inset-x-0">
      <div className="flex h-[55px] items-center justify-between p-4">
        <Link href="/" className="flex font-bold text-xl ml-4">
          <Image src={logo} height={10} width={30} alt="dump" /> &nbsp;
          <span>ExAI</span>
        </Link>

        <NavigationMenu className="ml-8 lg:mr-4">
          <NavigationMenuList>
            <NavigationMenuItem>
              <NavigationMenuTrigger className="w-[200px] lg:w-full">
                Explainable AI Algorithms
              </NavigationMenuTrigger>
              <NavigationMenuContent className="left-0">
                <ul className="grid gap-3 p-6 md:w-[400px] lg:w-[500px] lg:grid-cols-[.75fr_1fr]">
                  <li className="row-span-3">
                    <NavigationMenuLink asChild>
                      <a className="flex h-full w-full select-none flex-col justify-end rounded-md bg-gradient-to-b from-muted/50 to-muted p-6 no-underline outline-none shadow-md">
                        <iframe
                          src="https://giphy.com/embed/7VzgMsB6FLCilwS30v"
                          width="150"
                          height="150"
                          className="rounded-lg pointer-events-none	"
                        ></iframe>
                        <p className="text-sm leading-tight mt-2 text-muted-foreground">
                          Generative AI refers to deep-learning models that can
                          generate high-quality text, images, and other content
                          based on the data they were trained on.
                        </p>
                      </a>
                    </NavigationMenuLink>
                  </li>
                  <ListItem
                    href="/algorithms/lime"
                    className="duration-200"
                    title="LIME"
                  >
                    LIME, the acronym for local interpretable model-agnostic
                    explanations, is a technique that approximates any black box
                    machine learning model with a local, interpretable model to
                    explain each individual prediction.
                  </ListItem>
                  <ListItem
                    href="/algorithms/gradcam"
                    className="duration-200"
                    title="Grad-CAM"
                  >
                    Gradient-weighted Class Activation Mapping is a technique
                    used in deep learning to visualize and understand the
                    decisions made by a CNN.
                  </ListItem>
                  <ListItem
                    href="/algorithms/mfpp"
                    className="duration-200"
                    title="MFPP"
                  >
                    MFPP is an abbreviation for Morphological Fragmental
                    Perturbation Pyramid, which is a black-box interpretation
                    method for deep neural networks (DNNs).
                  </ListItem>
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>

            <NavigationMenuItem>
              <NavigationMenuTrigger>How it was done!</NavigationMenuTrigger>
              <NavigationMenuContent>
                <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px] ">
                  {components.map((component) => (
                    <ListItem
                      key={component.title}
                      title={component.title}
                      href={component.href}
                    >
                      {component.description}
                    </ListItem>
                  ))}
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>

        <div className="text-white">
          <Button className="w-[40px] rounded-xl shadow-lg hover:scale-110 transition duration-300">
            <Link href="https://github.com/prasadnakhate1705/final_year/">
              <Github />
            </Link>
          </Button>
        </div>
      </div>
    </nav>
  );
};

const ListItem = React.forwardRef<
  React.ElementRef<"a">,
  React.ComponentPropsWithoutRef<"a">
>(({ className, title, children, ...props }, ref) => {
  return (
    <li>
      <NavigationMenuLink asChild>
        <a
          ref={ref}
          className={cn(
            "block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
            className
          )}
          {...props}
        >
          <div className="text-sm font-medium leading-none">{title}</div>
          <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
            {children}
          </p>
        </a>
      </NavigationMenuLink>
    </li>
  );
});
ListItem.displayName = "ListItem";

export default Navbar;
