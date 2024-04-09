import { buttonVariants } from "@/components/ui/button";
import Link from "next/link";
import cn from "../lib/utils";

const Navbar = () => {
  return (
    <nav className="lex-no-wrap fixed top-0 backdrop-blur-sm h-[55px] w-full z-30 shadow-md shadow-black/5 dark:bg-neutral-600 dark:shadow-black/10 transition-all inset-x-0">
      <div className="flex h-[55px] items-center justify-between">
        <Link href="/" className="flex font-bold z-40 text-xl ml-4">
          <span>EXAI</span>
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
